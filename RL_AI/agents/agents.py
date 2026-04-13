"""Agents that operate on C# SeaEngine snapshots and action lists."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from RL_AI.SeaEngine.observation import ACTION_FEATURE_DIM, build_observation


class SeaEngineAgent(ABC):
    def __init__(self, name: str, seed: Optional[int] = None) -> None:
        self.name = name
        self.rng = random.Random(seed)

    @abstractmethod
    def select_action(
        self,
        snapshot: Dict[str, Any],
        legal_actions: Sequence[Dict[str, Any]],
    ) -> Tuple[int, Dict[str, Any]]:
        raise NotImplementedError


class SeaEngineRandomAgent(SeaEngineAgent):
    auto_play_strategy: str = "random"  # C# apply_auto에서 사용할 전략 이름

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__("random", seed=seed)

    def select_action(
        self,
        snapshot: Dict[str, Any],
        legal_actions: Sequence[Dict[str, Any]],
    ) -> Tuple[int, Dict[str, Any]]:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        idx = self.rng.randrange(len(legal_actions))
        return idx, legal_actions[idx]


class SeaEngineGreedyAgent(SeaEngineAgent):
    auto_play_strategy: str = "greedy"  # C# apply_auto에서 사용할 전략 이름

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__("greedy", seed=seed)

    def select_action(
        self,
        snapshot: Dict[str, Any],
        legal_actions: Sequence[Dict[str, Any]],
    ) -> Tuple[int, Dict[str, Any]]:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        scored = [(self._score_action(snapshot, action), idx, action) for idx, action in enumerate(legal_actions)]
        best_score = max(score for score, _, _ in scored)
        best = [(idx, action) for score, idx, action in scored if score == best_score]
        return best[self.rng.randrange(len(best))]

    def _score_action(self, snapshot: Dict[str, Any], action: Dict[str, Any]) -> int:
        effect_id = action.get("effect_id", "")
        target = action.get("target", {})
        target_type = target.get("type", "None")
        target_uid = target.get("guid", "")
        cards = {card["uid"]: card for card in snapshot.get("board", [])}
        target_card = cards.get(target_uid)

        score = 0
        if effect_id == "DefaultAttack":
            score += 80
        elif effect_id == "DeployUnit":
            score += 45
        elif effect_id == "DefaultMove":
            score += 20
        elif effect_id == "TurnEnd":
            score -= 100
        else:
            score += 55

        if target_type == "Unit" and target_card is not None:
            if target_card.get("role") == "Leader":
                score += 100
            score += 10 - min(int(target_card.get("hp", 10)), 10)

        if target_type == "Cell":
            target_x = int(target.get("pos_x", -1))
            target_y = int(target.get("pos_y", -1))
            enemy_leader = next(
                (card for card in snapshot.get("board", []) if card.get("owner") != snapshot.get("active_player") and card.get("role") == "Leader" and card.get("is_placed")),
                None,
            )
            if enemy_leader is not None:
                distance = abs(target_x - int(enemy_leader.get("pos_x", -1))) + abs(target_y - int(enemy_leader.get("pos_y", -1)))
                score += max(0, 8 - distance)

        score += self.rng.randint(0, 3)
        return score


@dataclass
class SeaEngineRLAgentOutput:
    action_index: int
    action: Dict[str, Any]
    state_vector: List[float]
    action_feature_vectors: List[List[float]]
    logits: List[float]
    probabilities: List[float]
    log_prob: float
    value: float


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_tensor: torch.Tensor, action_tensor: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        is_batched = state_tensor.dim() == 2
        if not is_batched:
            state_tensor = state_tensor.unsqueeze(0)
            action_tensor = action_tensor.unsqueeze(0)

        state_hidden = self.state_encoder(state_tensor)
        action_hidden = self.action_encoder(action_tensor)
        repeated_state = state_hidden.unsqueeze(1).expand(-1, action_tensor.size(1), -1)
        logits = self.policy_head(torch.cat([repeated_state, action_hidden], dim=-1)).squeeze(-1)
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)

        value = self.value_head(state_hidden).squeeze(-1)

        if not is_batched:
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        return logits, value


class SeaEngineRLAgent(SeaEngineAgent):
    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        sample_actions: bool = True,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__("rl", seed=seed)
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.sample_actions = sample_actions
        self.device = torch.device(device)
        self.state_dim: Optional[int] = None
        self.action_dim = ACTION_FEATURE_DIM
        self.model: Optional[PPOActorCritic] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.last_output: Optional[SeaEngineRLAgentOutput] = None

    @contextmanager
    def sampling_mode(self, enabled: bool):
        previous = self.sample_actions
        self.sample_actions = enabled
        try:
            yield
        finally:
            self.sample_actions = previous

    def ensure_model(self, state_dim: int) -> None:
        if self.model is not None:
            return
        self.state_dim = state_dim
        self.model = PPOActorCritic(state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def clone_as_opponent(
        self,
        *,
        name: Optional[str] = None,
        sample_actions: bool = False,
        seed: Optional[int] = None,
    ) -> "SeaEngineRLAgent":
        cloned = SeaEngineRLAgent(
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate,
            sample_actions=sample_actions,
            device=str(self.device),
            seed=seed,
        )
        if self.model is None or self.state_dim is None:
            return cloned
        cloned.ensure_model(self.state_dim)
        assert cloned.model is not None and self.model is not None
        cloned.model.load_state_dict(self.model.state_dict())
        cloned.model.eval()
        cloned.name = self.name if name is None else name
        return cloned

    def save(self, path: str | Path) -> None:
        """Saves physical model weights and hyperparameters."""
        if self.model is None or self.state_dim is None:
            raise RuntimeError("Cannot save an uninitialized model.")
        
        payload = {
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, path)
        
    def load(self, path: str | Path) -> None:
        """Loads physical model weights and hyperparameters."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.state_dim = payload["state_dim"]
        self.hidden_dim = payload["hidden_dim"]
        self.ensure_model(self.state_dim)
        assert self.model is not None
        self.model.load_state_dict(payload["state_dict"])

    def forward_tensors(self, state_vector: Sequence[float], action_vectors: Sequence[Sequence[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.ensure_model(len(state_vector))
        assert self.model is not None
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action_vectors, dtype=torch.float32, device=self.device)
        return self.model(state_tensor, action_tensor)

    def compute_policy_output(
        self,
        snapshot: Dict[str, Any],
        legal_actions: Sequence[Dict[str, Any]],
    ) -> SeaEngineRLAgentOutput:
        observation = build_observation({**snapshot, "actions": list(legal_actions)}, snapshot.get("active_player"))
        logits_tensor, value_tensor = self.forward_tensors(observation.state_vector, observation.action_feature_vectors)
        dist = Categorical(logits=logits_tensor)
        chosen_index = int(dist.sample().item()) if self.sample_actions else int(torch.argmax(logits_tensor).item())
        output = SeaEngineRLAgentOutput(
            action_index=chosen_index,
            action=legal_actions[chosen_index],
            state_vector=observation.state_vector,
            action_feature_vectors=observation.action_feature_vectors,
            logits=logits_tensor.detach().cpu().tolist(),
            probabilities=dist.probs.detach().cpu().tolist(),
            log_prob=float(dist.log_prob(torch.tensor(chosen_index, device=self.device)).item()),
            value=float(value_tensor.item()),
        )
        self.last_output = output
        return output

    def compute_policy_output_batched(
        self,
        snapshots: Sequence[Dict[str, Any]],
        legal_actions_list: Sequence[Sequence[Dict[str, Any]]],
    ) -> List[SeaEngineRLAgentOutput]:
        """배치 단위로 여러 상태의 정책 출력을 한 번에 계산합니다. (벡터화된 환경 지원용)"""
        if not snapshots:
            return []
            
        observations = [
            build_observation({**snap, "actions": list(acts)}, snap.get("active_player"))
            for snap, acts in zip(snapshots, legal_actions_list)
        ]
        
        state_vectors = [obs.state_vector for obs in observations]
        action_feature_vectors_list = [obs.action_feature_vectors for obs in observations]
        
        self.ensure_model(len(state_vectors[0]))
        assert self.model is not None
        
        batch_size = len(state_vectors)
        max_actions = max((len(avs) for avs in action_feature_vectors_list), default=0)
        
        if max_actions == 0:
            raise ValueError("No legal actions available in any of the batched snapshots.")
            
        padded_actions = torch.zeros((batch_size, max_actions, self.action_dim), dtype=torch.float32, device=self.device)
        action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool, device=self.device)
        
        for i, avs in enumerate(action_feature_vectors_list):
            num_actions = len(avs)
            if num_actions > 0:
                padded_actions[i, :num_actions, :] = torch.tensor(avs, dtype=torch.float32, device=self.device)
                action_mask[i, :num_actions] = True
                
        state_tensor = torch.tensor(state_vectors, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits, values = self.model(state_tensor, padded_actions, action_mask)
            
        outputs = []
        for i in range(batch_size):
            valid_logits = logits[i][:len(action_feature_vectors_list[i])]
            dist = Categorical(logits=valid_logits)
            chosen_index = int(dist.sample().item()) if self.sample_actions else int(torch.argmax(valid_logits).item())
            
            output = SeaEngineRLAgentOutput(
                action_index=chosen_index,
                action=legal_actions_list[i][chosen_index],
                state_vector=state_vectors[i],
                action_feature_vectors=action_feature_vectors_list[i],
                logits=valid_logits.detach().cpu().tolist(),
                probabilities=dist.probs.detach().cpu().tolist(),
                log_prob=float(dist.log_prob(torch.tensor(chosen_index, device=self.device)).item()),
                value=float(values[i].item()),
            )
            outputs.append(output)
            
        if outputs:
            self.last_output = outputs[-1]
            
        return outputs

    def select_action(
        self,
        snapshot: Dict[str, Any],
        legal_actions: Sequence[Dict[str, Any]],
    ) -> Tuple[int, Dict[str, Any]]:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        output = self.compute_policy_output(snapshot, legal_actions)
        return output.action_index, output.action

    def evaluate_action_set(
        self,
        state_vector: Sequence[float],
        action_feature_vectors: Sequence[Sequence[float]],
        chosen_action_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward_tensors(state_vector, action_feature_vectors)
        dist = Categorical(logits=logits)
        action_index_tensor = torch.tensor(chosen_action_index, dtype=torch.long, device=self.device)
        log_prob = dist.log_prob(action_index_tensor)
        entropy = dist.entropy()
        return log_prob, entropy, value

    def evaluate_action_set_batched(
        self,
        state_vectors: List[List[float]],
        action_feature_vectors_list: List[List[List[float]]],
        chosen_action_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.ensure_model(len(state_vectors[0]))
        assert self.model is not None
        
        batch_size = len(state_vectors)
        max_actions = max(len(avs) for avs in action_feature_vectors_list)
        
        padded_actions = torch.zeros((batch_size, max_actions, self.action_dim), dtype=torch.float32, device=self.device)
        action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool, device=self.device)
        
        for i, avs in enumerate(action_feature_vectors_list):
            num_actions = len(avs)
            if num_actions > 0:
                padded_actions[i, :num_actions, :] = torch.tensor(avs, dtype=torch.float32, device=self.device)
                action_mask[i, :num_actions] = True
                
        state_tensor = torch.tensor(state_vectors, dtype=torch.float32, device=self.device)
        
        logits, value = self.model(state_tensor, padded_actions, action_mask)
        
        dist = Categorical(logits=logits)
        action_index_tensor = torch.tensor(chosen_action_indices, dtype=torch.long, device=self.device)
        log_prob = dist.log_prob(action_index_tensor)
        entropy = dist.entropy()
        
        return log_prob, entropy, value
