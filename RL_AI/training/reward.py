"""Advanced terminal reward for SeaEngine.
Designed to prevent reward hacking while encouraging tactical advantage and efficiency.
"""

from __future__ import annotations
from typing import Dict, Any, Optional

def terminal_reward_for_player(
    result: str, 
    player_id: str, 
    final_snapshot: Optional[Dict[str, Any]] = None
) -> float:
    """게임 종료 시의 보상을 정밀하게 계산합니다."""
    if result in {"Draw", "Ongoing"}:
        return 0.0
        
    is_winner = (result == "Player1Win" and player_id == "P1") or \
                (result == "Player2Win" and player_id == "P2")
    
    # 1. 기본 승패 보상 (가장 큰 비중)
    reward = 1.0 if is_winner else -1.0
    
    if final_snapshot:
        # 2. 효율성 보너스 (빨리 이길수록 좋음, 최대 +0.2)
        turn = final_snapshot.get("turn", 100)
        if is_winner:
            reward += max(0, (50 - turn) * 0.004)
        else:
            # 지더라도 오래 버티면 패널티 경감 (최대 +0.1)
            reward += min(0.1, turn * 0.001)

        # 3. 우위 보너스 (상대 리더와의 HP 차이 반영, 종료 시점에만 계산하여 해킹 방지)
        p1_hp = 0
        p2_hp = 0
        for card in final_snapshot.get("board", []):
            if card.get("role") == "Leader" and card.get("is_placed"):
                if card.get("owner") == "P1": p1_hp = card.get("hp", 0)
                else: p2_hp = card.get("hp", 0)
        
        hp_delta = (p1_hp - p2_hp) if player_id == "P1" else (p2_hp - p1_hp)
        reward += (hp_delta * 0.02) # HP 1점당 0.02 보너스/패널티

    return float(reward)
