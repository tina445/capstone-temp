"""Checkpoint training experiment for SeaEngine (Asynchronous version).
Fixed to save models in RL_AI/models/ and only save the RL agent.
"""

from __future__ import annotations

import os
import time
import datetime
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any

import torch

from RL_AI.agents.agents import SeaEngineAgent, SeaEngineGreedyAgent, SeaEngineRLAgent, SeaEngineRandomAgent
from RL_AI.training.trainer import SeaEnginePPOTrainer
from RL_AI.analysis.reports import build_win_rate_report, save_report


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def zip_and_cleanup_logs(log_dir: Path, prefix: str):
    """새로 생성된 로그들을 압축하고 원본을 삭제합니다."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = log_dir / f"logs_{prefix}_{timestamp}.zip"
    
    # 압축 대상 파일 목록 (최근 1시간 내 생성된 txt, jsonl)
    now = time.time()
    log_files = []
    for ext in ["*.txt", "*.jsonl"]:
        for p in log_dir.glob(ext):
            if now - p.stat().st_mtime < 3600: # 1시간 이내
                log_files.append(p)
    
    if not log_files:
        return
        
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in log_files:
            zipf.write(f, f.name)
            
    for f in log_files:
        try:
            os.remove(f)
        except:
            pass
    print(f"\n[Logs Archived] => {zip_path}")


async def run_checkpoint_training_experiment(
    agent: SeaEngineRLAgent,
    train_opponent_pool: Optional[Sequence[SeaEngineAgent]] = None,
    eval_greedy_agent: Optional[SeaEngineAgent] = None,
    eval_random_agent: Optional[SeaEngineAgent] = None,
    eval_matches: int = 100,
    total_train_episodes: int = 3000,
    eval_interval: int = 600,
    max_turns: int = 100,
    update_interval: int = 16,
    self_play_snapshot_interval_updates: int = 5,
    max_self_play_opponents: int = 2,
    alternate_player_sides: bool = True,
    card_data_path: Optional[str] = None,
    player1_deck: str = "",
    player2_deck: str = "",
    seed: Optional[int] = None,
    summary_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """[Pure Async Experiment] Runs the training loop using await."""
    start_time = time.time()
    log_dir = Path("RL_AI/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    device = _auto_device()
    print(f"Using device: {device}  |  CUDA available: {torch.cuda.is_available()}")
    agent.device = torch.device(device)

    trainer = SeaEnginePPOTrainer(agent)
    greedy_eval_opponent = eval_greedy_agent or SeaEngineGreedyAgent()
    random_eval_opponent = eval_random_agent or SeaEngineRandomAgent()

    checkpoints: List[Dict[str, object]] = []
    best_random_checkpoint: Optional[Dict[str, object]] = None
    
    summary_lines = [
        "=== SeaEngine Checkpoint Training Experiment ===",
        f"total_train_episodes={total_train_episodes}",
        f"eval_interval={eval_interval}",
        "",
    ]

    # --- Initial Evaluation ---
    before_greedy = await trainer.evaluate(
        opponent_agent=greedy_eval_opponent,
        num_matches=eval_matches,
        card_data_path=card_data_path,
        player1_deck=player1_deck,
        player2_deck=player2_deck,
        max_turns=max_turns,
    )
    summary_lines.extend([
        "=== Before Training vs Greedy ===",
        f"report={before_greedy['report_path']}",
        build_win_rate_report(before_greedy),
        "",
    ])

    episodes_completed = 0
    while episodes_completed < total_train_episodes:
        current_batch = min(eval_interval, total_train_episodes - episodes_completed)
        print(f"\n--- Training Episodes {episodes_completed + 1} to {episodes_completed + current_batch} ---")
        
        train_stats = await trainer.train(
            num_episodes=current_batch,
            num_envs=8,
            opponent_pool=train_opponent_pool,
            card_data_path=card_data_path,
            player1_deck=player1_deck,
            player2_deck=player2_deck,
            max_turns=max_turns,
            update_interval=update_interval,
            self_play_snapshot_interval_updates=self_play_snapshot_interval_updates,
            max_self_play_opponents=max_self_play_opponents,
            alternate_player_sides=alternate_player_sides,
            progress_callback=lambda ep, tot, opp, st: print(
                f"[train ep={episodes_completed + ep}/{total_train_episodes}] opponent={opp} | "
                f"w/l/d={st.get('wins')}/{st.get('losses')}/{st.get('draws')} | updates={st.get('updates')}"
            ) if (episodes_completed + ep) % 200 == 0 else None
        )
        
        episodes_completed += current_batch
        
        eval_greedy = await trainer.evaluate(
            opponent_agent=greedy_eval_opponent,
            num_matches=eval_matches,
            card_data_path=card_data_path,
            player1_deck=player1_deck,
            player2_deck=player2_deck,
            max_turns=max_turns,
        )
        eval_random = await trainer.evaluate(
            opponent_agent=random_eval_opponent,
            num_matches=eval_matches,
            card_data_path=card_data_path,
            player1_deck=player1_deck,
            player2_deck=player2_deck,
            max_turns=max_turns,
        )

        checkpoint_data = {
            "episodes": episodes_completed,
            "greedy_stats": eval_greedy,
            "random_stats": eval_random,
            "train_summary": train_stats,
        }
        checkpoints.append(checkpoint_data)

        summary_lines.extend([
            f"=== Checkpoint {episodes_completed} Episodes ===",
            f"greedy_report={eval_greedy['report_path']}",
            f"random_report={eval_random['report_path']}",
            "",
        ])

        models_dir = Path("RL_AI/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        if best_random_checkpoint is None or eval_random["p1_wins"] > best_random_checkpoint["random_stats"]["p1_wins"]:
            best_random_checkpoint = checkpoint_data
            agent.save(models_dir / "my_best_model.pt")

    # --- Final Summary ---
    summary_lines.extend([
        "=== Best Checkpoint Summary ===",
        f"episodes_completed={best_random_checkpoint['episodes'] if best_random_checkpoint else 0}",
    ])

    final_report = "\n".join(summary_lines)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = summary_report_path or f"RL_AI/log/training_report_{ts}.txt"
    save_report(final_report, report_file)

    total_time_sec = time.time() - start_time
    
    # 실험 종료 후 로그 압축 및 정리
    zip_and_cleanup_logs(log_dir, "train")

    return {
        "checkpoints": checkpoints, 
        "best_random": best_random_checkpoint,
        "total_time_sec": total_time_sec,
        "summary_report_path": str(report_file)
    }
