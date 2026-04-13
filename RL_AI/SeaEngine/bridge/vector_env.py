"""Vectorized environment for running multiple SeaEngine sessions in parallel.
Robust version that doesn't hold fixed loop references to avoid context errors.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

from RL_AI.SeaEngine.bridge.seaengine_session import SeaEngineSession
from RL_AI.agents.agents import SeaEngineRLAgent, SeaEngineAgent
from RL_AI.training.storage import RolloutBuffer, RolloutStep # 경로 수정됨


class VectorSeaEngineEnv:
    def __init__(
        self,
        num_envs: int,
        card_data_path: Optional[str] = None,
        player1_deck: str = "",
        player2_deck: str = "",
        max_turns: int = 100,
    ) -> None:
        self.num_envs = num_envs
        self.card_data_path = card_data_path
        self.player1_deck = player1_deck
        self.player2_deck = player2_deck
        self.max_turns = max_turns
        
        self.sessions = [SeaEngineSession(card_data_path=card_data_path) for _ in range(num_envs)]
        self.executor = ThreadPoolExecutor(max_workers=num_envs)
        
        self.current_snapshots: List[Optional[Dict[str, Any]]] = [None] * num_envs
        self.active_mask = [False] * num_envs
        self.episode_ids = [0] * num_envs
        self.learning_player_ids = ["P1"] * num_envs

    def start(self) -> None:
        for session in self.sessions:
            session.start()

    def close(self) -> None:
        for session in self.sessions:
            session.close()
        self.executor.shutdown(wait=True)

    async def _run_in_executor(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        if kwargs:
            func = partial(func, *args, **kwargs)
            args = []
        return await loop.run_in_executor(self.executor, func, *args)

    async def reset_all(self, learning_player_ids: List[str], episode_start_id: int) -> List[Dict[str, Any]]:
        tasks = []
        for i, session in enumerate(self.sessions):
            self.learning_player_ids[i] = learning_player_ids[i]
            self.episode_ids[i] = episode_start_id + i
            tasks.append(self._run_in_executor(
                session.init_game, 
                player1_deck=self.player1_deck, 
                player2_deck=self.player2_deck
            ))
        
        self.current_snapshots = await asyncio.gather(*tasks)
        self.active_mask = [True] * self.num_envs
        return self.current_snapshots

    async def step(
        self, 
        agent: SeaEngineRLAgent, 
        opponent_pool: List[SeaEngineAgent],
        buffer: RolloutBuffer
    ) -> Tuple[int, int]:
        rl_indices = []
        rl_snapshots = []
        rl_legal_actions = []
        
        manual_opp_indices = []
        manual_opp_actions = []

        for i in range(self.num_envs):
            if not self.active_mask[i]:
                continue
                
            snap = self.current_snapshots[i]
            if snap["result"] != "Ongoing" or snap["turn"] > self.max_turns:
                self.active_mask[i] = False
                continue
                
            active_player = snap["active_player"]
            legal_actions = snap.get("actions", [])
            
            if not legal_actions:
                self.active_mask[i] = False
                continue

            if active_player == self.learning_player_ids[i]:
                rl_indices.append(i)
                rl_snapshots.append(snap)
                rl_legal_actions.append(legal_actions)
            else:
                opp = opponent_pool[i % len(opponent_pool)]
                from RL_AI.SeaEngine.action_adapter import choose_action_with_agent
                _, action = choose_action_with_agent(opp, snap)
                manual_opp_indices.append(i)
                manual_opp_actions.append(action)

        if rl_indices:
            outputs = agent.compute_policy_output_batched(rl_snapshots, rl_legal_actions)
            
            apply_tasks = []
            for i, idx in enumerate(rl_indices):
                output = outputs[i]
                buffer.add_step(RolloutStep(
                    episode_id=self.episode_ids[idx],
                    player_id=0 if self.learning_player_ids[idx] == "P1" else 1,
                    state_vector=output.state_vector,
                    action_feature_vectors=output.action_feature_vectors,
                    chosen_action_index=output.action_index,
                    reward=0.0,
                    done=False,
                    old_log_prob=output.log_prob,
                    old_value=output.value,
                ))
                
                opp = opponent_pool[idx % len(opponent_pool)]
                strategy = getattr(opp, "auto_play_strategy", None)
                
                if strategy:
                    apply_tasks.append(self._run_in_executor(
                        self.sessions[idx].apply_and_auto_play,
                        action_uid=output.action["uid"],
                        learner_id=self.learning_player_ids[idx],
                        strategy=strategy
                    ))
                else:
                    apply_tasks.append(self._run_in_executor(
                        self.sessions[idx].apply_action,
                        action_uid=output.action["uid"]
                    ))
            
            rl_results = await asyncio.gather(*apply_tasks)
            for i, idx in enumerate(rl_indices):
                self.current_snapshots[idx] = rl_results[i]

        if manual_opp_indices:
            opp_tasks = [
                self._run_in_executor(self.sessions[idx].apply_action, action_uid=action["uid"])
                for idx, action in zip(manual_opp_indices, manual_opp_actions)
            ]
            opp_results = await asyncio.gather(*opp_tasks)
            for i, idx in enumerate(manual_opp_indices):
                self.current_snapshots[idx] = opp_results[i]

        return sum(self.active_mask), len(rl_indices)

    def _assign_terminal_rewards_for_session(self, buffer: RolloutBuffer, episode_id: int, player_idx: int, result: str, final_snapshot: Dict[Any, Any]):
        from RL_AI.training.reward import terminal_reward_for_player
        player_id = "P1" if player_idx == 0 else "P2"
        terminal_reward = terminal_reward_for_player(result, player_id, final_snapshot)
        
        last_step_idx = -1
        for i in reversed(range(len(buffer.steps))):
            if buffer.steps[i].episode_id == episode_id and buffer.steps[i].player_id == player_idx:
                last_step_idx = i
                break
        
        if last_step_idx != -1:
            buffer.steps[last_step_idx].reward = terminal_reward
            buffer.steps[last_step_idx].done = True

    async def collect_data(
        self, 
        agent: SeaEngineRLAgent, 
        opponent_pool: List[SeaEngineAgent], 
        total_episodes: int,
        buffer: RolloutBuffer
    ) -> Dict[str, Any]:
        episodes_started = 0
        episodes_finished = 0
        total_steps = 0
        stats = {"wins": 0, "losses": 0, "draws": 0}
        
        learning_player_ids = ["P1" if i % 2 == 0 else "P2" for i in range(self.num_envs)]
        await self.reset_all(learning_player_ids, episodes_started)
        episodes_started += self.num_envs
        
        while episodes_finished < total_episodes:
            active_count, rl_step_count = await self.step(agent, opponent_pool, buffer)
            total_steps += rl_step_count
            
            reset_indices = []
            reset_learning_ids = []
            
            for i in range(self.num_envs):
                snap = self.current_snapshots[i]
                if self.active_mask[i] and (snap["result"] != "Ongoing" or snap["turn"] > self.max_turns):
                    res = str(snap["result"])
                    player_id = self.learning_player_ids[i]
                    
                    if (player_id == "P1" and res == "Player1Win") or (player_id == "P2" and res == "Player2Win"):
                        stats["wins"] += 1
                    elif res in {"Player1Win", "Player2Win"}:
                        stats["losses"] += 1
                    else:
                        stats["draws"] += 1

                    self._assign_terminal_rewards_for_session(
                        buffer, self.episode_ids[i], 0 if player_id == "P1" else 1, res, snap
                    )
                    
                    episodes_finished += 1
                    self.active_mask[i] = False
                    
                    if episodes_started < total_episodes:
                        reset_indices.append(i)
                        reset_learning_ids.append("P1" if episodes_started % 2 == 0 else "P2")
                        episodes_started += 1
            
            if reset_indices:
                tasks = [
                    self._run_in_executor(
                        self.sessions[idx].init_game, 
                        player1_deck=self.player1_deck, 
                        player2_deck=self.player2_deck
                    ) for idx in reset_indices
                ]
                results = await asyncio.gather(*tasks)
                for i, idx in enumerate(reset_indices):
                    self.current_snapshots[idx] = results[i]
                    self.learning_player_ids[idx] = reset_learning_ids[i]
                    self.episode_ids[idx] = episodes_started - len(reset_indices) + i
                    self.active_mask[idx] = True
                    
            if active_count == 0 and episodes_started >= total_episodes:
                break
                
        return {
            "episodes": episodes_finished,
            "total_steps": total_steps,
            **stats
        }
