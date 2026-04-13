"""Evaluator for SeaEngine agents with both side/deck matrix and same-deck matchups.
Unified Asynchronous version for reliable Jupyter/DLPC execution.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from RL_AI.SeaEngine.bridge.vector_env import VectorSeaEngineEnv
from RL_AI.agents.agents import SeaEngineAgent, SeaEngineRLAgent
from RL_AI.training.storage import RolloutBuffer
from RL_AI.analysis.reports import build_win_rate_report, save_report


async def evaluate_agents_async(
    agent: SeaEngineRLAgent,
    opponent: SeaEngineAgent,
    num_matches: int = 20,
    card_data_path: Optional[str] = None,
    player1_deck: str = "",
    player2_deck: str = "",
    max_turns: int = 100,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
) -> Dict[str, Any]:
    env = VectorSeaEngineEnv(
        num_envs=min(8, num_matches),
        card_data_path=card_data_path,
        player1_deck=player1_deck,
        player2_deck=player2_deck,
        max_turns=max_turns
    )
    env.start()
    buffer = RolloutBuffer()
    try:
        with agent.sampling_mode(False):
            stats = await env.collect_data(agent, [opponent], num_matches, buffer)
        
        results = {
            "episodes": stats["episodes"],
            "p1_wins": stats["wins"],
            "p2_wins": stats["losses"],
            "draws": stats["draws"],
            "avg_steps": stats["total_steps"] / stats["episodes"] if stats["episodes"] > 0 else 0,
            "avg_final_turn": 0.0,
            "p1_agent": agent.name,
            "p2_agent": opponent.name,
            "action_type_counts": {},
            "card_use_counts": {}
        }
        
        report_text = build_win_rate_report(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("RL_AI/log")
        log_dir.mkdir(parents=True, exist_ok=True)
        report_path = log_dir / f"eval_{timestamp}.txt"
        save_report(report_text, str(report_path))
        results["report_path"] = str(report_path)
        return results
    finally:
        env.close()


async def evaluate_agents(agent, opponent, **kwargs):
    return await evaluate_agents_async(agent, opponent, **kwargs)


class SeaEngineComprehensiveEvaluator:
    def __init__(self, agent: SeaEngineRLAgent, card_data_path: Optional[str] = None, num_envs: int = 8):
        self.agent = agent
        self.card_data_path = card_data_path
        self.num_envs = num_envs

    async def run_scenario(self, opponent, rl_side, p1_deck_name, p2_deck_name, num_matches=100):
        OR_DECK = '["Or_L", "Or_B", "Or_N", "Or_R", "Or_P", "Or_P", "Or_P"]'
        CL_DECK = '["Cl_L", "Cl_B", "Cl_N", "Cl_R", "Cl_P", "Cl_P", "Cl_P"]'
        
        p1_deck = OR_DECK if p1_deck_name == "Orange" else CL_DECK
        p2_deck = OR_DECK if p2_deck_name == "Orange" else CL_DECK
            
        env = VectorSeaEngineEnv(
            num_envs=min(self.num_envs, num_matches),
            card_data_path=self.card_data_path,
            player1_deck=p1_deck,
            player2_deck=p2_deck,
            max_turns=100
        )
        env.start()
        buffer = RolloutBuffer()
        try:
            stats = {"episodes": 0, "wins": 0, "losses": 0, "draws": 0, "total_steps": 0}
            with self.agent.sampling_mode(False):
                remaining = num_matches
                while remaining > 0:
                    batch = min(remaining, 25)
                    sub_stats = await env.collect_data(self.agent, [opponent], batch, buffer)
                    for k in stats: stats[k] += sub_stats[k]
                    remaining -= sub_stats["episodes"]
                    print(f"    [{stats['episodes']}/{num_matches}] W/L/D: {stats['wins']}/{stats['losses']}/{stats['draws']}")
            return stats
        finally:
            env.close()

    async def run_full_matrix(self, opponents: List[SeaEngineAgent], num_matches_per_case: int = 100) -> Dict[str, Any]:
        results = {}
        for opp in opponents:
            opp_name = opp.name
            results[opp_name] = {}
            # 1. 덱 스왑 매치업 (8개 조합)
            for side in ["P1", "P2"]:
                for rl_deck in ["Orange", "Charlotte"]:
                    opp_deck = "Charlotte" if rl_deck == "Orange" else "Orange"
                    p1_deck = rl_deck if side == "P1" else opp_deck
                    p2_deck = opp_deck if side == "P1" else rl_deck
                    
                    case_name = f"{side}_{rl_deck}_vs_{opp_deck}"
                    print(f"Evaluating: {opp_name} | {side}({rl_deck}) vs Opp({opp_deck})")
                    stats = await self.run_scenario(opp, side, p1_deck, p2_deck, num_matches_per_case)
                    results[opp_name][case_name] = self._format_stats(stats)

            # 2. 동일 덱 매치업 (8개 조합 추가)
            for side in ["P1", "P2"]:
                for same_deck in ["Orange", "Charlotte"]:
                    case_name = f"{side}_{same_deck}_vs_{same_deck}"
                    print(f"Evaluating: {opp_name} | {side}({same_deck}) vs Opp({same_deck}) [Mirror]")
                    stats = await self.run_scenario(opp, side, same_deck, same_deck, num_matches_per_case)
                    results[opp_name][case_name] = self._format_stats(stats)
        return results

    def _format_stats(self, stats):
        return {
            "win_rate": (stats["wins"] / stats["episodes"]) * 100 if stats["episodes"] > 0 else 0,
            "wins": stats["wins"], "losses": stats["losses"], "draws": stats["draws"],
            "avg_steps": stats["total_steps"] / stats["episodes"] if stats["episodes"] > 0 else 0
        }


def print_matrix_report(matrix_results: Dict[str, Any]):
    print("\n" + "="*70)
    print(" [SeaEngine RL Agent Comprehensive 16-Way Matrix Report] ")
    print("="*70)
    for opp_name, cases in matrix_results.items():
        print(f"\nTarget Opponent: {opp_name}")
        print("-" * 65)
        print(f"{'Condition (RL vs Opp)':<35} | {'Win Rate':<10} | {'W/L/D':<10}")
        print("-" * 65)
        for case_name, data in cases.items():
            display_name = case_name.replace("_", " ")
            print(f"{display_name:<35} | {data['win_rate']:>8.1f}% | {data['wins']}/{data['losses']}/{data['draws']}")
    print("="*70)
