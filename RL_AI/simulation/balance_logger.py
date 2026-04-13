import sys
import json
from pathlib import Path
from typing import Dict, Any

def format_pos(x: int, y: int) -> str:
    """Format x,y to board coordinates, e.g. 0,2 -> 1C"""
    if x < 0 or y < 0:
        return "??"
    return f"{x+1}{chr(ord('A') + y)}"

def format_board(board: list) -> str:
    # Sort board to group by Owner, then Role, then x, y
    sorted_board = sorted(board, key=lambda c: (c["owner"], c["role"] != "Leader", c["pos_x"], c["pos_y"]))
    cards = []
    for c in sorted_board:
        pos_str = format_pos(c["pos_x"], c["pos_y"])
        hp_str = f"({c['hp']}/{c['max_hp']})"
        cards.append(f"{c['owner']} {c['name']}@{pos_str}{hp_str}")
    return "; ".join(cards)

def action_to_text(action: Dict[str, Any], before: Dict[str, Any]) -> str:
    board = {c["uid"]: c for c in before.get("board", [])}
    
    src_uid = action.get("source", "")
    src_card = board.get(src_uid)
    src_name = src_card["name"] if src_card else src_uid

    effect_id = action.get("effect_id", "")
    target = action.get("target", {})
    t_type = target.get("type", "None")
    t_guid = target.get("guid", "")
    t_x = target.get("pos_x", -1)
    t_y = target.get("pos_y", -1)

    t_card = board.get(t_guid)
    t_name = f"{t_card['owner']} {t_card['name']}" if t_card else t_guid
    pos_str = format_pos(t_x, t_y)

    if effect_id == "DeployUnit":
        return f"{src_name} 소환 -> {pos_str}"
    elif effect_id == "DefaultMove":
        if src_card:
            src_name = f"{src_card['owner']} {src_name}"
        return f"{src_name} 이동 -> {pos_str}"
    elif effect_id == "DefaultAttack":
        if src_card:
            src_name = f"{src_card['owner']} {src_name}"
        return f"{src_name} 공격 -> {t_name}"
    elif effect_id == "TurnEnd":
        return "턴 종료"
    else:
        # Skill usage
        if src_card and src_card.get("role") != "Leader":
             # In SeaEngine, cards cast skills based on their own name
             skill_src = src_name
        else:
             skill_src = src_name
             
        t_str = f"대상={t_name}" if t_type == "Unit" else f"위치={pos_str}"
        return f"{skill_src} 사용 | {t_str}"

def process_file(in_path: Path, out_path: Path):
    with open(in_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            
            try:
                match = json.loads(line)
            except json.JSONDecodeError:
                print(f"Error parsing JSON on line {line_idx+1}")
                continue

            fout.write("=== MATCH START ===\n")
            events = match.get("events", [])
            if events:
                first = events[0]["before"]
                fout.write(f"turn={first['turn']} active={first['active_player']} phase={first['phase']} result={first['result']}\n")
                fout.write(f"P1 leader_hp={first['p1_hp']} | P2 leader_hp={first['p2_hp']}\n")

            for evt_idx, evt in enumerate(events):
                before = evt["before"]
                after = evt["after"]
                action = evt["action"]
                
                t = before["turn"]
                p = before["active_player"]
                ph = before["phase"]
                
                act_str = action_to_text(action, before)
                
                board_str = format_board(after["board"])
                res = after["result"]
                p1h = after["p1_hp"]
                p2h = after["p2_hp"]

                fout.write(f"T{t} {p} {ph} | {act_str}\n")
                fout.write(f"    -> result={res} P1_HP={p1h} P2_HP={p2h} board={board_str} | note=step={evt_idx}\n")

            fout.write("=== MATCH END ===\n")
            fout.write(f"turn={match.get('final_turn', '?')} result={match.get('final_result', 'Unknown')}\n\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python balance_logger.py <path_to_raw_jsonl>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"File not found: {in_path}")
        sys.exit(1)

    out_path = in_path.with_suffix(".txt")
    process_file(in_path, out_path)
    print(f"Human-readable log successfully generated: {out_path}")

if __name__ == "__main__":
    main()
