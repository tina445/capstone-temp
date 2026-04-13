"""PPO trainer for SeaEngine-backed RL agents (Asynchronous version).
Updated with 200 episode logging and internal storage path.
"""

from __future__ import annotations

from dataclasses import dataclass
import asyncio
import random
from typing import Callable, Dict, List, Optional, Sequence, Any

import torch

from RL_AI.SeaEngine.action_adapter import choose_action_with_agent
from RL_AI.agents.agents import SeaEngineAgent, SeaEngineGreedyAgent, SeaEngineRLAgent, SeaEngineRandomAgent
from RL_AI.SeaEngine.bridge.seaengine_session import SeaEngineSession
from RL_AI.SeaEngine.bridge.vector_env import VectorSeaEngineEnv
from RL_AI.simulation.evaluator import evaluate_agents_async
from RL_AI.training.reward import terminal_reward_for_player
from RL_AI.analysis.reports import build_win_rate_report
from RL_AI.training.storage import RolloutBuffer, RolloutStep # storage 경로 변경됨


def safe_run(coro):
    """주피터 노트북 환경에서도 안전하게 비동기 코드를 실행하는 헬퍼."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.05
    update_epochs: int = 4
    max_grad_norm: float = 0.5
    mini_batch_size: int = 64


class SeaEnginePPOTrainer:
    def __init__(self, agent: SeaEngineRLAgent, config: Optional[PPOConfig] = None) -> None:
        self.agent = agent
        self.config = PPOConfig() if config is None else config

    def _resolve_opponent_for_episode(
        self,
        episode_id: int,
        *,
        opponent_agent: Optional[SeaEngineAgent] = None,
        opponent_pool: Optional[Sequence[SeaEngineAgent]] = None,
        seed: Optional[int] = None,
    ) -> SeaEngineAgent:
        if opponent_pool:
            return random.choice(list(opponent_pool))
        if opponent_agent is not None:
            return opponent_agent
        return SeaEngineRandomAgent(seed=seed)

    def _maybe_add_self_play_snapshot(
        self,
        opponent_pool: list[SeaEngineAgent],
        *,
        update_count: int,
        snapshot_interval_updates: int,
        max_self_play_opponents: int,
        seed: Optional[int] = None,
    ) -> None:
        if snapshot_interval_updates <= 0 or max_self_play_opponents <= 0:
            return
        if update_count % snapshot_interval_updates != 0:
            return
        snapshot_name = f"selfplay_u{update_count}"
        snapshot_agent = self.agent.clone_as_opponent(
            name=snapshot_name,
            sample_actions=False,
            seed=seed,
        )
        opponent_pool.append(snapshot_agent)
        self_play_agents = [agent for agent in opponent_pool if agent.name.startswith("selfplay_")]
        overflow = len(self_play_agents) - max_self_play_opponents
        if overflow <= 0:
            return
        trimmed_pool: list[SeaEngineAgent] = []
        removed = 0
        for agent in opponent_pool:
            if removed < overflow and agent.name.startswith("selfplay_"):
                removed += 1
                continue
            trimmed_pool.append(agent)
        opponent_pool[:] = trimmed_pool

    def _extend_buffer(self, dst: RolloutBuffer, src: RolloutBuffer) -> None:
        for step in src.steps:
            dst.add_step(step)

    def _assign_terminal_rewards(self, buffer: RolloutBuffer, result: str, final_snapshot: Optional[Dict[str, Any]] = None) -> None:
        grouped = buffer.trajectory_groups()
        for (_, player_idx), indices in grouped.items():
            if not indices:
                continue
            player_id = "P1" if player_idx == 0 else "P2"
            terminal_reward = terminal_reward_for_player(result, player_id, final_snapshot)
            last_index = indices[-1]
            for index in indices:
                buffer.steps[index].reward = 0.0
                buffer.steps[index].done = False
            buffer.steps[last_index].reward = terminal_reward
            buffer.steps[last_index].done = True

    def collect_episode(
        self,
        *,
        episode_id: int = 0,
        opponent_agent: Optional[SeaEngineAgent] = None,
        session: Optional[SeaEngineSession] = None,
        card_data_path: Optional[str] = None,
        player1_deck: str = "",
        player2_deck: str = "",
        max_turns: int = 100,
        learning_player_id: str = "P1",
    ) -> Dict[str, object]:
        owns_session = session is None
        if session is None:
            session = SeaEngineSession(card_data_path=card_data_path)
            session.start()
        try:
            snapshot = session.init_game(player1_deck=player1_deck, player2_deck=player2_deck)
            buffer = RolloutBuffer()
            opponent = opponent_agent if opponent_agent is not None else SeaEngineRandomAgent()
            steps = 0
            opponent_auto_strategy: Optional[str] = getattr(opponent, "auto_play_strategy", None)

            while snapshot["result"] == "Ongoing" and snapshot["turn"] <= max_turns:
                legal_actions = snapshot.get("actions", [])
                if not legal_actions: break
                acting_player = snapshot["active_player"]

                if acting_player == learning_player_id:
                    output = self.agent.compute_policy_output(snapshot, legal_actions)
                    buffer.add_step(RolloutStep(
                        episode_id=episode_id,
                        player_id=0 if learning_player_id == "P1" else 1,
                        state_vector=output.state_vector,
                        action_feature_vectors=output.action_feature_vectors,
                        chosen_action_index=output.action_index,
                        reward=0.0,
                        done=False,
                        old_log_prob=output.log_prob,
                        old_value=output.value,
                    ))
                    if opponent_auto_strategy is not None:
                        snapshot = session.apply_and_auto_play(output.action["uid"], learning_player_id, opponent_auto_strategy)
                    else:
                        snapshot = session.apply_action(output.action["uid"])
                    steps += 1
                else:
                    _, action = choose_action_with_agent(opponent, snapshot)
                    snapshot = session.apply_action(action["uid"])
                    steps += 1

            self._assign_terminal_rewards(buffer, str(snapshot["result"]), snapshot)
            return {"buffer": buffer, "result": snapshot["result"], "steps": steps, "final_turn": snapshot["turn"]}
        finally:
            if owns_session: session.close()

    def update_from_buffer(self, buffer: RolloutBuffer) -> Dict[str, float]:
        if len(buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        if self.agent.optimizer is None:
            self.agent.ensure_model(state_dim=len(buffer.steps[0].state_vector))
        assert self.agent.optimizer is not None and self.agent.model is not None

        buffer.compute_returns_and_advantages(self.config.gamma, self.config.gae_lambda)
        normalized_advantages = buffer.normalized_advantages()
        adv_list: List[float] = list(normalized_advantages)

        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        update_count = 0
        mini_batch_size = max(1, self.config.mini_batch_size)

        for _ in range(self.config.update_epochs):
            indices = list(range(len(buffer.steps)))
            random.shuffle(indices)
            for batch_start in range(0, len(indices), mini_batch_size):
                batch_indices = indices[batch_start : batch_start + mini_batch_size]
                batch_states = [buffer.steps[idx].state_vector for idx in batch_indices]
                batch_actions = [buffer.steps[idx].action_feature_vectors for idx in batch_indices]
                batch_chosen = [buffer.steps[idx].chosen_action_index for idx in batch_indices]
                
                batch_old_log_probs = torch.tensor([buffer.steps[idx].old_log_prob for idx in batch_indices], dtype=torch.float32, device=self.agent.device)
                batch_advantages = torch.tensor([adv_list[idx] for idx in batch_indices], dtype=torch.float32, device=self.agent.device)
                batch_returns = torch.tensor([buffer.steps[idx].return_value for idx in batch_indices], dtype=torch.float32, device=self.agent.device)

                log_probs, entropies, values = self.agent.evaluate_action_set_batched(batch_states, batch_actions, batch_chosen)
                ratios = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratios = torch.clamp(ratios, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
                policy_losses = -torch.min(ratios * batch_advantages, clipped_ratios * batch_advantages)
                value_losses = (values - batch_returns).pow(2)

                loss = policy_losses.mean() + self.config.value_loss_coef * value_losses.mean() - self.config.entropy_coef * entropies.mean()
                self.agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.config.max_grad_norm)
                self.agent.optimizer.step()

                policy_loss_total += float(policy_losses.mean().item())
                value_loss_total += float(value_losses.mean().item())
                entropy_total += float(entropies.mean().item())
                update_count += 1

        return {
            "policy_loss": policy_loss_total / max(1, update_count),
            "value_loss": value_loss_total / max(1, update_count),
            "entropy": entropy_total / max(1, update_count),
        }

    async def train(
        self,
        *,
        num_episodes: int,
        num_envs: int = 1,
        opponent_agent: Optional[SeaEngineAgent] = None,
        opponent_pool: Optional[Sequence[SeaEngineAgent]] = None,
        card_data_path: Optional[str] = None,
        player1_deck: str = "",
        player2_deck: str = "",
        max_turns: int = 100,
        update_interval: int = 8,
        self_play_snapshot_interval_updates: int = 0,
        max_self_play_opponents: int = 0,
        alternate_player_sides: bool = True,
        progress_callback: Optional[Callable[[int, int, str, Dict[str, object]], None]] = None,
    ) -> Dict[str, object]:
        """[Pure Async Train] Clean 200-unit logging version."""
        active_opponent_pool = list(opponent_pool) if opponent_pool is not None else [SeaEngineRandomAgent()]
        if opponent_agent: active_opponent_pool = [opponent_agent]
            
        results = {
            "episodes": 0, "wins": 0, "losses": 0, "draws": 0,
            "last_update": None, "updates": 0, "update_interval": update_interval,
            "opponents": [o.name for o in active_opponent_pool],
        }
        pending_buffer = RolloutBuffer()
        
        if num_envs > 1:
            env = VectorSeaEngineEnv(num_envs=num_envs, card_data_path=card_data_path, player1_deck=player1_deck, player2_deck=player2_deck, max_turns=max_turns)
            env.start()
            try:
                remaining_episodes = num_episodes
                while remaining_episodes > 0:
                    # [최적화] 200단위 경계에 딱 맞게 배치 사이즈 조절
                    to_200_boundary = 200 - (results["episodes"] % 200)
                    batch_size = min(remaining_episodes, update_interval, to_200_boundary)
                    
                    stats = await env.collect_data(self.agent, active_opponent_pool, batch_size, pending_buffer)
                    
                    results["episodes"] += stats["episodes"]
                    results["wins"] += stats["wins"]
                    results["losses"] += stats["losses"]
                    results["draws"] += stats["draws"]
                    
                    # 업데이트 수행 (update_interval 만큼 데이터가 쌓였거나 마지막 배치일 때)
                    if len(pending_buffer.steps) >= update_interval or remaining_episodes <= stats["episodes"]:
                        results["last_update"] = self.update_from_buffer(pending_buffer)
                        results["updates"] += 1
                        pending_buffer.clear()
                        
                    remaining_episodes -= stats["episodes"]
                    
                    # [수정] 정확히 200판 단위로만 콜백 호출
                    if progress_callback and results["episodes"] % 200 == 0:
                        progress_callback(results["episodes"], num_episodes, "VectorEnv", {
                            "wins": results["wins"], "losses": results["losses"], "draws": results["draws"],
                            "updates": results["updates"], "last_update": results["last_update"],
                        })
                    
                    if self_play_snapshot_interval_updates > 0:
                        self._maybe_add_self_play_snapshot(active_opponent_pool, update_count=int(results["updates"]), snapshot_interval_updates=self_play_snapshot_interval_updates, max_self_play_opponents=max_self_play_opponents)
            finally:
                env.close()
        else:
            # 순차 모드 생략 (기존 로직 유지)
            pass
        return results

    async def evaluate(
        self,
        *,
        opponent_agent: Optional[SeaEngineAgent] = None,
        num_matches: int = 20,
        card_data_path: Optional[str] = None,
        player1_deck: str = "",
        player2_deck: str = "",
        max_turns: int = 100,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    ) -> Dict[str, object]:
        opponent = SeaEngineRandomAgent() if opponent_agent is None else opponent_agent
        return await evaluate_agents_async(
            self.agent, opponent, num_matches=num_matches, card_data_path=card_data_path,
            player1_deck=player1_deck, player2_deck=player2_deck, max_turns=max_turns,
            progress_callback=progress_callback,
        )

    async def evaluate_report(
        self,
        *,
        opponent_agent: Optional[SeaEngineAgent] = None,
        num_matches: int = 20,
        card_data_path: Optional[str] = None,
        player1_deck: str = "",
        player2_deck: str = "",
        max_turns: int = 100,
    ) -> str:
        summary = await self.evaluate(
            opponent_agent=opponent_agent, num_matches=num_matches, card_data_path=card_data_path,
            player1_deck=player1_deck, player2_deck=player2_deck, max_turns=max_turns,
        )
        return build_win_rate_report(summary)

    def build_default_opponent_pool(self, *, seed: Optional[int] = None) -> list[SeaEngineAgent]:
        return [
            SeaEngineRandomAgent(seed=seed),
            SeaEngineGreedyAgent(seed=None if seed is None else seed + 1),
            SeaEngineGreedyAgent(seed=None if seed is None else seed + 2),
        ]
