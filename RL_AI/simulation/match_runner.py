import argparse
import sys
import json
import random
import time
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Dict

# RL_AI module path resolution
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch

from RL_AI.SeaEngine.bridge.seaengine_session import SeaEngineSession
from RL_AI.SeaEngine.action_adapter import choose_action_with_agent
from RL_AI.agents.agents import (
    SeaEngineAgent,
    SeaEngineRandomAgent,
    SeaEngineGreedyAgent,
    SeaEngineRLAgent,
)

class HumanTerminalAgent(SeaEngineAgent):
    def __init__(self, name: str = "human"):
        super().__init__(name)

    def select_action(self, snapshot: Dict[str, Any], legal_actions: list[Dict[str, Any]]) -> tuple[int, Dict[str, Any]]:
        print(f"\n--- Turn {snapshot['turn']}, Phase {snapshot['phase']}, {snapshot['active_player']} ---")
        print("Board:")
        for card in snapshot.get("board", []):
            if card.get("is_placed"):
               print(f"  [{card['owner']}] {card.get('name', card.get('card_id'))} HP:{card.get('hp')}/{card.get('max_hp')} @ ({card.get('pos_x')},{card.get('pos_y')})")
        print("\nLegal Actions:")
        for i, a in enumerate(legal_actions):
            text = a.get("text", str(a))
            print(f"  {i}: {text}")

        while True:
            try:
                idx = int(input("Select action idx: "))
                if 0 <= idx < len(legal_actions):
                    return idx, legal_actions[idx]
            except ValueError:
                pass
            print("Invalid index.")

def get_agent_by_type(agent_type: str, model_path: str = "") -> SeaEngineAgent:
    if agent_type == "human":
        return HumanTerminalAgent()
    elif agent_type == "random":
        return SeaEngineRandomAgent()
    elif agent_type == "greedy":
        return SeaEngineGreedyAgent()
    elif agent_type == "rl_untrained":
        return SeaEngineRLAgent(device="cuda" if torch.cuda.is_available() else "cpu", sample_actions=False)
    elif agent_type == "rl_trained":
        if not model_path:
            raise ValueError("model_path is required for rl_trained agent")
        agent = SeaEngineRLAgent(device="cuda" if torch.cuda.is_available() else "cpu", sample_actions=False)
        agent.load(model_path)
        return agent
    raise ValueError(f"Unknown agent type: {agent_type}")

def condense_snapshot(snap: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "turn": snap.get("turn", 0),
        "active_player": snap.get("active_player", ""),
        "phase": snap.get("phase", ""),
        "result": snap.get("result", ""),
        "board": [
            {
               "uid": c["uid"], 
               "owner": c["owner"], 
               "name": str(c.get("name", c.get("card_id"))), 
               "role": c.get("role", "none"),
               "hp": c.get("hp", 0), 
               "max_hp": c.get("max_hp", 0),
               "pos_x": c.get("pos_x", -1), 
               "pos_y": c.get("pos_y", -1)
            } for c in snap.get("board", []) if c.get("is_placed")
        ],
        "p1_hp": next((c.get("hp", 0) for c in snap.get("board", []) if c.get("owner")=="P1" and c.get("role")=="Leader" and c.get("is_placed")), 0),
        "p2_hp": next((c.get("hp", 0) for c in snap.get("board", []) if c.get("owner")=="P2" and c.get("role")=="Leader" and c.get("is_placed")), 0)
    }

def main():
    parser = argparse.ArgumentParser(description="Run SeaEngine matches and record raw logs.")
    parser.add_argument("--p1", type=str, default="rl_untrained", choices=["human", "random", "greedy", "rl_untrained", "rl_trained"])
    parser.add_argument("--p2", type=str, default="greedy", choices=["human", "random", "greedy", "rl_untrained", "rl_trained"])
    parser.add_argument("--p1_model", type=str, default="", help="Path to P1 model if p1 is rl_trained")
    parser.add_argument("--p2_model", type=str, default="", help="Path to P2 model if p2 is rl_trained")
    parser.add_argument("--matches", type=int, default=1, help="Number of matches to run per configuration")
    parser.add_argument("--max_turns", type=int, default=100, help="Max turns per game")
    parser.add_argument("--deck1", type=str, default="", help="P1 deck string")
    parser.add_argument("--deck2", type=str, default="", help="P2 deck string")
    parser.add_argument("--comprehensive", action="store_true", help="Run full cross-validation grid (800 matches recommended)")
    args = parser.parse_args()

    OR_DECK = '["Or_L", "Or_B", "Or_N", "Or_R", "Or_P", "Or_P", "Or_P"]'
    CL_DECK = '["Cl_L", "Cl_B", "Cl_N", "Cl_R", "Cl_P", "Cl_P", "Cl_P"]'

    configs = []
    if args.comprehensive:
        print("=== Comprehensive Grid Evaluation ===")
        args.matches = 100 if args.matches == 1 else args.matches # Default to 100 per grid if not specified
        target_agent = "rl_trained" if args.p1_model else "rl_untrained"
        
        for opp in ["random", "greedy"]:
            for rl_pos in ["P1", "P2"]:
                for rl_deck_name in ["Orange", "Clear"]:
                    p1_type = target_agent if rl_pos == "P1" else opp
                    p2_type = opp if rl_pos == "P1" else target_agent
                    
                    deck1_name = rl_deck_name if rl_pos == "P1" else ("Clear" if rl_deck_name == "Orange" else "Orange")
                    deck2_name = rl_deck_name if rl_pos == "P2" else ("Clear" if rl_deck_name == "Orange" else "Orange")
                    
                    d1_str = OR_DECK if deck1_name == "Orange" else CL_DECK
                    d2_str = OR_DECK if deck2_name == "Orange" else CL_DECK
                    
                    configs.append({
                        "p1_type": p1_type, "p2_type": p2_type,
                        "d1_str": d1_str, "d2_str": d2_str,
                        "desc": f"RL({rl_pos}, {rl_deck_name}) vs {opp.capitalize()}"
                    })
    else:
        configs.append({
            "p1_type": args.p1, "p2_type": args.p2,
            "d1_str": args.deck1, "d2_str": args.deck2,
            "desc": f"P1 ({args.p1}) vs P2 ({args.p2})"
        })

    log_dir = Path(__file__).resolve().parent / "log"
    log_dir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"raw_match_{ts}.jsonl"

    total_matches = len(configs) * args.matches
    print(f"Total Configurations: {len(configs)}")
    print(f"Matches per config: {args.matches}")
    print(f"Total Matches to run: {total_matches}")
    print(f"Saving raw logs to {log_path}")

    session = SeaEngineSession()
    session.start()
    
    agent_cache = {}
    def get_cached_agent(atype, is_p1):
        key = f"{atype}_{is_p1}"
        if key not in agent_cache:
            model_path = args.p1_model if is_p1 else args.p2_model
            if args.comprehensive and "model" not in key and "trained" in atype:
                model_path = args.p1_model or args.p2_model
            ag = get_agent_by_type(atype, model_path)
            ag.name = atype
            agent_cache[key] = ag
        return agent_cache[key]

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            match_counter = 0
            for cfg in configs:
                print(f"\n[Config] {cfg['desc']}")
                
                p1_ag = get_cached_agent(cfg["p1_type"], True)
                p2_ag = get_cached_agent(cfg["p2_type"], False)
                p1_ctx = p1_ag.sampling_mode(False) if hasattr(p1_ag, "sampling_mode") else nullcontext()
                p2_ctx = p2_ag.sampling_mode(False) if hasattr(p2_ag, "sampling_mode") else nullcontext()

                with p1_ctx, p2_ctx:
                    for _ in range(args.matches):
                        match_counter += 1
                        snapshot = session.init_game(player1_deck=cfg["d1_str"], player2_deck=cfg["d2_str"])
                        agents = {"P1": p1_ag, "P2": p2_ag}
                        
                        match_record = {
                            "match": match_counter,
                            "p1": cfg["p1_type"],
                            "p2": cfg["p2_type"],
                            "config": cfg["desc"],
                            "events": []
                        }
                        
                        while snapshot["result"] == "Ongoing" and snapshot["turn"] <= args.max_turns:
                            actions = snapshot.get("actions", [])
                            if not actions: break

                            acting_agent = agents[snapshot["active_player"]]
                            _, action = choose_action_with_agent(acting_agent, snapshot)

                            evt = {"before": condense_snapshot(snapshot), "action": action}
                            snapshot = session.apply_action(action["uid"])
                            evt["after"] = condense_snapshot(snapshot)
                            match_record["events"].append(evt)

                        match_record["final_result"] = snapshot["result"]
                        match_record["final_turn"] = snapshot["turn"]
                        f.write(json.dumps(match_record, ensure_ascii=False) + "\n")
                        
                        if match_counter % 20 == 0 or match_counter == total_matches:
                            print(f"  [{cfg['desc']}] {match_counter}/{total_matches} matches completed...")

    finally:
        session.close()
    
    print("\nAll matches finished!")
    print(f"To generate human-readable file: python RL_AI/simulation/balance_logger.py {log_path}")

if __name__ == "__main__":
    main()
