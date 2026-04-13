"""Observation and action feature builders for the C# SeaEngine snapshot.

Performance notes
-----------------
build_observation() creates a ``_Ctx`` object once from the raw snapshot dict,
pre-computing all expensive lookups (board-by-uid dict, placed-card lists,
leader positions, adjacent-enemy counts).  All internal helpers receive this
context so that repeated O(N) board scans are eliminated.

Public API is unchanged: build_observation(), encode_action_features(),
build_fixed_state_vector(), ACTION_FEATURE_DIM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

BOARD_SIZE = 6
MAX_BOARD_CARDS = 14
MAX_HAND_CARDS = 7

ROLE_ORDER = ["Leader", "Bishop", "Knight", "Rook", "Pawn"]
ROLE_BY_SUFFIX = {
    "L": "Leader",
    "B": "Bishop",
    "N": "Knight",
    "R": "Rook",
    "P": "Pawn",
}
EFFECT_BUCKETS = ["DeployUnit", "DefaultMove", "DefaultAttack", "TurnEnd", "Skill"]
TARGET_BUCKETS = ["None", "Cell", "Unit", "Unit2", "Card"]
RESULT_BUCKETS = ["Ongoing", "Player1Win", "Player2Win", "Draw"]

def _flip_pos_dict(d: Dict[str, Any]) -> None:
    if "pos_x" in d and isinstance(d["pos_x"], int) and d["pos_x"] >= 0:
        d["pos_x"] = 5 - d["pos_x"]
    if "pos_y" in d and isinstance(d["pos_y"], int) and d["pos_y"] >= 0:
        d["pos_y"] = 5 - d["pos_y"]

def _canonicalize_snapshot(snap: Dict[str, Any], player_id: str) -> Dict[str, Any]:
    if player_id == "P1":
        return snap
    import copy
    s = copy.deepcopy(snap)
    for c in s.get("board", []):
        _flip_pos_dict(c)
    for a in s.get("actions", []):
        if "target" in a and isinstance(a["target"], dict):
            _flip_pos_dict(a["target"])
    return s

def _canonicalize_action(action: Dict[str, Any], player_id: str) -> Dict[str, Any]:
    if player_id == "P1":
        return action
    import copy
    a = copy.deepcopy(action)
    if "target" in a and isinstance(a["target"], dict):
        _flip_pos_dict(a["target"])
    return a



# ---------------------------------------------------------------------------
# Tiny normalisation helpers (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class SeaEngineObservation:
    unit_list: List[Dict[str, Any]]
    hand_list: List[Dict[str, Any]]
    global_vector: List[float]
    legal_action_mask: List[int]
    state_vector: List[float]
    action_feature_vectors: List[List[float]]


def _normalize_ratio(value: float, scale: float) -> float:
    return 0.0 if scale == 0 else value / scale


def _normalize_pos(value: int) -> float:
    if value < 0:
        return -1.0
    return value / float(BOARD_SIZE - 1)


def _distance(x1: int, y1: int, x2: int, y2: int) -> float:
    if min(x1, y1, x2, y2) < 0:
        return -1.0
    return (abs(x1 - x2) + abs(y1 - y2)) / float((BOARD_SIZE - 1) * 2)


def _role_one_hot(role: str) -> List[float]:
    return [1.0 if role == name else 0.0 for name in ROLE_ORDER]


def _effect_one_hot(effect_id: str) -> List[float]:
    bucket = effect_id if effect_id in EFFECT_BUCKETS[:-1] else "Skill"
    return [1.0 if bucket == name else 0.0 for name in EFFECT_BUCKETS]


def _target_one_hot(target_type: str) -> List[float]:
    return [1.0 if target_type == name else 0.0 for name in TARGET_BUCKETS]


def _result_one_hot(result: str) -> List[float]:
    return [1.0 if result == name else 0.0 for name in RESULT_BUCKETS]


def _role_from_card(card: Dict[str, Any]) -> str:
    role = str(card.get("role", ""))
    if role:
        return role
    return _role_from_card_id(str(card.get("card_id", "")))


def _role_from_card_id(card_id: str) -> str:
    suffix = card_id.split("_")[-1].strip()[-1:] if card_id else ""
    return ROLE_BY_SUFFIX.get(suffix, "")


def _status_summary(card: Dict[str, Any]) -> tuple[float, float, float, float]:
    attack_mod = 0.0
    has_move_lock = 0.0
    has_attack_lock = 0.0
    timed_status_count = 0.0
    for status in card.get("statuses", []):
        timed_status_count += 1.0
        status_type = status.get("type", "")
        if status_type == "AttackModifier":
            attack_mod += float(status.get("value", 0.0))
        elif status_type == "MoveLock":
            has_move_lock = 1.0
        elif status_type == "AttackLock":
            has_attack_lock = 1.0
    return attack_mod, has_move_lock, has_attack_lock, timed_status_count


# ---------------------------------------------------------------------------
# Pre-computed snapshot context  ← 핵심 캐싱 구조체
# ---------------------------------------------------------------------------

@dataclass
class _Ctx:
    """스냅샷에서 비싼 조회를 미리 계산해 둔 컨텍스트.

    build_observation() 진입 시 한 번만 생성되고 모든 내부 함수에 전달된다.
    이를 통해 O(N) 보드 탐색 수백 회 → O(1) 딕셔너리 조회로 최적화된다.
    """
    player_id: str
    enemy_id: str
    board: List[Dict[str, Any]]
    board_by_uid: Dict[str, Dict[str, Any]]   # uid → card (O(1) 조회)
    placed: List[Dict[str, Any]]              # is_placed == True 카드 목록
    own_placed: List[Dict[str, Any]]          # 자기 편 배치 카드
    enemy_placed: List[Dict[str, Any]]        # 적 배치 카드
    own_player: Dict[str, Any]
    enemy_player: Dict[str, Any]
    own_leader: Optional[Dict[str, Any]]
    enemy_leader: Optional[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    actions_by_source: Dict[str, List[Dict[str, Any]]]
    # uid → 인접 적 카드 수 (attack_targets, enemy_neighbors, attackers_of 세 함수 대체)
    enemy_adj: Dict[str, float]


def _make_ctx(snapshot: Dict[str, Any], player_id: str) -> _Ctx:
    """스냅샷 dict에서 _Ctx를 생성한다.  O(N) — 단 한 번만 호출한다."""
    enemy_id = "P2" if player_id == "P1" else "P1"
    board: List[Dict[str, Any]] = snapshot.get("board", [])
    board_by_uid: Dict[str, Dict[str, Any]] = {str(c.get("uid", "")): c for c in board}

    placed = [c for c in board if c.get("is_placed")]
    own_placed = [c for c in placed if c.get("owner") == player_id]
    enemy_placed = [c for c in placed if c.get("owner") != player_id]

    players_by_id: Dict[str, Dict[str, Any]] = {
        str(p.get("id", "")): p for p in snapshot.get("players", [])
    }
    own_player = players_by_id.get(player_id, {})
    enemy_player = players_by_id.get(enemy_id, {})

    own_leader: Optional[Dict[str, Any]] = next(
        (c for c in own_placed if _role_from_card(c) == "Leader"), None
    )
    enemy_leader: Optional[Dict[str, Any]] = next(
        (c for c in enemy_placed if _role_from_card(c) == "Leader"), None
    )

    actions: List[Dict[str, Any]] = snapshot.get("actions", [])
    actions_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for action in actions:
        src = str(action.get("source", ""))
        actions_by_source.setdefault(src, []).append(action)

    # 위치 → 소유자 맵 (인접 적 카드 수 계산용)
    pos_to_owner: Dict[tuple[int, int], str] = {}
    for c in placed:
        x, y = int(c.get("pos_x", -1)), int(c.get("pos_y", -1))
        if x >= 0 and y >= 0:
            pos_to_owner[(x, y)] = str(c.get("owner", ""))

    # 카드별 인접 적 수 사전 계산 (O(N×8) → 이후 O(1))
    enemy_adj: Dict[str, float] = {}
    for c in board:
        uid = str(c.get("uid", ""))
        if not c.get("is_placed"):
            enemy_adj[uid] = 0.0
            continue
        owner = str(c.get("owner", ""))
        x, y = int(c.get("pos_x", -1)), int(c.get("pos_y", -1))
        if x < 0 or y < 0:
            enemy_adj[uid] = 0.0
            continue
        cnt = 0.0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nb_owner = pos_to_owner.get((x + dx, y + dy))
                if nb_owner is not None and nb_owner != owner:
                    cnt += 1.0
        enemy_adj[uid] = cnt

    return _Ctx(
        player_id=player_id,
        enemy_id=enemy_id,
        board=board,
        board_by_uid=board_by_uid,
        placed=placed,
        own_placed=own_placed,
        enemy_placed=enemy_placed,
        own_player=own_player,
        enemy_player=enemy_player,
        own_leader=own_leader,
        enemy_leader=enemy_leader,
        actions=actions,
        actions_by_source=actions_by_source,
        enemy_adj=enemy_adj,
    )


# ---------------------------------------------------------------------------
# Legacy public helpers (kept for backward compatibility / outside callers)
# ---------------------------------------------------------------------------

def _find_card(snapshot: Dict[str, Any], uid: str) -> Optional[Dict[str, Any]]:
    """Legacy O(N) lookup — 내부 코드는 ctx.board_by_uid 사용."""
    for card in snapshot.get("board", []):
        if card.get("uid") == uid:
            return card
    return None


def _actions_by_source(snapshot: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for action in snapshot.get("actions", []):
        source_uid = str(action.get("source", ""))
        mapping.setdefault(source_uid, []).append(action)
    return mapping


def _get_players(
    snapshot: Dict[str, Any], player_id: str
) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    enemy_id = "P2" if player_id == "P1" else "P1"
    players = {player["id"]: player for player in snapshot.get("players", [])}
    return players.get(player_id, {}), players.get(enemy_id, {}), enemy_id


def _get_leaders(
    snapshot: Dict[str, Any], player_id: str, enemy_id: str
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    own = None
    enemy = None
    for card in snapshot.get("board", []):
        if not card.get("is_placed"):
            continue
        if _role_from_card(card) != "Leader":
            continue
        if card.get("owner") == player_id:
            own = card
        elif card.get("owner") == enemy_id:
            enemy = card
    return own, enemy


def _count_ready_attack_targets(snapshot: Dict[str, Any], card: Dict[str, Any]) -> float:
    """Legacy — 내부 코드는 ctx.enemy_adj 사용."""
    if not card.get("is_placed"):
        return 0.0
    source_owner = card.get("owner")
    sx = int(card.get("pos_x", -1))
    sy = int(card.get("pos_y", -1))
    if sx < 0 or sy < 0:
        return 0.0
    reachable = 0.0
    for other in snapshot.get("board", []):
        if not other.get("is_placed") or other.get("owner") == source_owner:
            continue
        ox = int(other.get("pos_x", -1))
        oy = int(other.get("pos_y", -1))
        if ox < 0 or oy < 0:
            continue
        if abs(sx - ox) <= 1 and abs(sy - oy) <= 1:
            reachable += 1.0
    return reachable


def _count_enemy_neighbors(snapshot: Dict[str, Any], card: Dict[str, Any]) -> float:
    """Legacy — 내부 코드는 ctx.enemy_adj 사용."""
    return _count_ready_attack_targets(snapshot, card)


def _count_attackers_of_card(snapshot: Dict[str, Any], target_card: Dict[str, Any]) -> float:
    """Legacy — 내부 코드는 ctx.enemy_adj 사용 (거리 계산은 대칭적)."""
    return _count_ready_attack_targets(snapshot, target_card)


def _sorted_board_cards(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    role_rank = {name: idx for idx, name in enumerate(ROLE_ORDER)}
    return sorted(
        snapshot.get("board", []),
        key=lambda card: (
            card.get("owner", ""),
            role_rank.get(_role_from_card(card), 99),
            card.get("uid", ""),
        ),
    )


# ---------------------------------------------------------------------------
# Internal vector builders (ctx-aware)
# ---------------------------------------------------------------------------

def _build_global_vector(snapshot: Dict[str, Any], player_id: str, ctx: Optional[_Ctx] = None) -> List[float]:
    if ctx is None:
        ctx = _make_ctx(snapshot, player_id)

    own_player = ctx.own_player
    enemy_player = ctx.enemy_player
    own_leader = ctx.own_leader
    enemy_leader = ctx.enemy_leader
    own_board = ctx.own_placed
    enemy_board = ctx.enemy_placed
    actions = ctx.actions

    own_attack_total = sum(float(c.get("effective_atk", 0.0)) for c in own_board)
    enemy_attack_total = sum(float(c.get("effective_atk", 0.0)) for c in enemy_board)
    own_hp_total = sum(float(c.get("hp", 0.0)) for c in own_board)
    enemy_hp_total = sum(float(c.get("hp", 0.0)) for c in enemy_board)
    own_ready_move = sum(1.0 for c in own_board if not c.get("is_moved"))
    own_ready_attack = sum(1.0 for c in own_board if not c.get("is_attacked"))
    enemy_ready_attack = sum(1.0 for c in enemy_board if not c.get("is_attacked"))

    # 배치 가능 / 스킬 사용 가능 카드 수
    hand_uids = {str(c.get("uid", "")) for c in own_player.get("hand", [])}
    deploy_sources: set[str] = set()
    skill_sources: set[str] = set()
    own_attack_count = 0.0
    own_move_count = 0.0
    for action in actions:
        eid = str(action.get("effect_id", ""))
        src = str(action.get("source", ""))
        src_card = ctx.board_by_uid.get(src)
        if src_card is not None and src_card.get("owner") != player_id:
            continue
        if eid == "DeployUnit" and src in hand_uids:
            deploy_sources.add(src)
        elif eid not in {"DeployUnit", "DefaultMove", "DefaultAttack", "TurnEnd"} and src in hand_uids:
            skill_sources.add(src)
        elif eid == "DefaultAttack":
            own_attack_count += 1.0
        elif eid == "DefaultMove":
            own_move_count += 1.0

    own_deployable = float(len(deploy_sources))
    own_skill_actions = float(len(skill_sources))

    # 리더 위협 카운트 (ctx.enemy_adj 재사용)
    enemy_attackers_on_leader = (
        0.0 if own_leader is None
        else ctx.enemy_adj.get(str(own_leader.get("uid", "")), 0.0)
    )
    own_attackers_on_enemy_leader = (
        0.0 if enemy_leader is None
        else ctx.enemy_adj.get(str(enemy_leader.get("uid", "")), 0.0)
    )

    center_control_own = sum(
        1.0 for c in own_board if 2 <= int(c.get("pos_x", -1)) <= 3 and 2 <= int(c.get("pos_y", -1)) <= 3
    )
    center_control_enemy = sum(
        1.0 for c in enemy_board if 2 <= int(c.get("pos_x", -1)) <= 3 and 2 <= int(c.get("pos_y", -1)) <= 3
    )

    action_counts = {bucket: 0.0 for bucket in EFFECT_BUCKETS}
    for action in actions:
        effect_id = str(action.get("effect_id", ""))
        bucket = effect_id if effect_id in EFFECT_BUCKETS[:-1] else "Skill"
        action_counts[bucket] += 1.0
    action_total = max(1.0, float(len(actions)))

    return [
        _normalize_ratio(float(snapshot.get("turn", 1)), 100.0),
        1.0 if snapshot.get("active_player") == player_id else 0.0,
        *_result_one_hot(str(snapshot.get("result", "Ongoing"))),
        _normalize_ratio(float(own_player.get("hand_count", 0)), MAX_HAND_CARDS),
        _normalize_ratio(float(enemy_player.get("hand_count", 0)), MAX_HAND_CARDS),
        _normalize_ratio(float(own_player.get("deck_count", 0)), MAX_BOARD_CARDS),
        _normalize_ratio(float(enemy_player.get("deck_count", 0)), MAX_BOARD_CARDS),
        _normalize_ratio(float(own_player.get("trash_count", 0)), MAX_BOARD_CARDS),
        _normalize_ratio(float(enemy_player.get("trash_count", 0)), MAX_BOARD_CARDS),
        0.0 if own_leader is None else _normalize_ratio(float(own_leader.get("hp", 0)), max(1.0, float(own_leader.get("max_hp", 1)))),
        0.0 if enemy_leader is None else _normalize_ratio(float(enemy_leader.get("hp", 0)), max(1.0, float(enemy_leader.get("max_hp", 1)))),
        0.0 if own_leader is None or enemy_leader is None else _normalize_ratio(
            float(own_leader.get("hp", 0)) - float(enemy_leader.get("hp", 0)),
            max(1.0, float(own_leader.get("max_hp", 1))),
        ),
        _normalize_ratio(float(len(own_board)), MAX_BOARD_CARDS),
        _normalize_ratio(float(len(enemy_board)), MAX_BOARD_CARDS),
        _normalize_ratio(float(len(own_board) - len(enemy_board)), MAX_BOARD_CARDS),
        _normalize_ratio(own_attack_total, 40.0),
        _normalize_ratio(enemy_attack_total, 40.0),
        _normalize_ratio(own_attack_total - enemy_attack_total, 40.0),
        _normalize_ratio(own_hp_total, 40.0),
        _normalize_ratio(enemy_hp_total, 40.0),
        _normalize_ratio(own_ready_move, MAX_BOARD_CARDS),
        _normalize_ratio(own_ready_attack, MAX_BOARD_CARDS),
        _normalize_ratio(enemy_ready_attack, MAX_BOARD_CARDS),
        _normalize_ratio(own_deployable, MAX_HAND_CARDS),
        _normalize_ratio(own_skill_actions, MAX_HAND_CARDS),
        _normalize_ratio(own_attack_count, 20.0),
        _normalize_ratio(own_move_count, 20.0),
        _normalize_ratio(enemy_attackers_on_leader, 6.0),
        _normalize_ratio(own_attackers_on_enemy_leader, 6.0),
        _normalize_ratio(center_control_own, 4.0),
        _normalize_ratio(center_control_enemy, 4.0),
        *[_normalize_ratio(action_counts[bucket], action_total) for bucket in EFFECT_BUCKETS],
    ]


def _build_board_vector(snapshot: Dict[str, Any], player_id: str, ctx: Optional[_Ctx] = None) -> List[float]:
    if ctx is None:
        ctx = _make_ctx(snapshot, player_id)

    own_leader = ctx.own_leader
    enemy_leader = ctx.enemy_leader
    own_lx = int(own_leader.get("pos_x", -1)) if own_leader else -1
    own_ly = int(own_leader.get("pos_y", -1)) if own_leader else -1
    enemy_lx = int(enemy_leader.get("pos_x", -1)) if enemy_leader else -1
    enemy_ly = int(enemy_leader.get("pos_y", -1)) if enemy_leader else -1

    # _sorted_board_cards 인라인 (ctx.board 재사용)
    role_rank = {name: idx for idx, name in enumerate(ROLE_ORDER)}
    cards = sorted(
        ctx.board,
        key=lambda c: (c.get("owner", ""), role_rank.get(_role_from_card(c), 99), c.get("uid", "")),
    )

    vectors: List[float] = []
    for card in cards[:MAX_BOARD_CARDS]:
        uid = str(card.get("uid", ""))
        attack_mod, has_move_lock, has_attack_lock, timed_status_count = _status_summary(card)
        cx = int(card.get("pos_x", -1))
        cy = int(card.get("pos_y", -1))
        role = _role_from_card(card)
        hp = float(card.get("hp", 0.0))
        max_hp = max(1.0, float(card.get("max_hp", 1.0)))
        effective_atk = float(card.get("effective_atk", 0.0))
        base_atk = float(card.get("atk", 0.0))

        # ctx.enemy_adj로 O(1) 조회 (기존 O(N) 탐색 3회 대체)
        reachable_targets = ctx.enemy_adj.get(uid, 0.0)
        adjacent_enemies = ctx.enemy_adj.get(uid, 0.0)
        incoming_attackers = ctx.enemy_adj.get(uid, 0.0)

        card_actions = ctx.actions_by_source.get(uid, [])
        has_attack_action = 1.0 if any(str(a.get("effect_id", "")) in {"DefaultAttack", "PawnGeneric"} for a in card_actions) else 0.0
        has_move_action = 1.0 if any(str(a.get("effect_id", "")) == "DefaultMove" for a in card_actions) else 0.0
        dist_enemy_leader = _distance(cx, cy, enemy_lx, enemy_ly)
        threatens_enemy_leader = 1.0 if enemy_leader is not None and 0.0 <= dist_enemy_leader <= (1.0 / 5.0) else 0.0
        in_center = 1.0 if 2 <= cx <= 3 and 2 <= cy <= 3 else 0.0
        row_progress = 0.0
        if card.get("is_placed"):
            row_progress = _normalize_ratio(float(cx if card.get("owner") == player_id else (BOARD_SIZE - 1 - cx)), BOARD_SIZE - 1)

        vectors.extend([
            1.0 if card.get("owner") == player_id else -1.0,
            1.0 if card.get("is_placed") else 0.0,
            1.0 if card.get("is_moved") else 0.0,
            1.0 if card.get("is_attacked") else 0.0,
            _normalize_pos(cx),
            _normalize_pos(cy),
            _normalize_ratio(hp, max_hp),
            _normalize_ratio(max_hp, 10.0),
            _normalize_ratio(base_atk, 10.0),
            _normalize_ratio(effective_atk, 10.0),
            _normalize_ratio(attack_mod, 5.0),
            has_move_lock,
            has_attack_lock,
            _normalize_ratio(timed_status_count, 4.0),
            _distance(cx, cy, own_lx, own_ly),
            dist_enemy_leader,
            _normalize_ratio(reachable_targets, 6.0),
            _normalize_ratio(adjacent_enemies, 6.0),
            _normalize_ratio(incoming_attackers, 6.0),
            has_attack_action,
            has_move_action,
            threatens_enemy_leader,
            in_center,
            row_progress,
            *_role_one_hot(role),
        ])

    missing_slots = MAX_BOARD_CARDS - min(len(cards), MAX_BOARD_CARDS)
    if missing_slots > 0:
        vectors.extend([0.0] * missing_slots * 29)
    return vectors


def _build_hand_vector(snapshot: Dict[str, Any], player_id: str, ctx: Optional[_Ctx] = None) -> List[float]:
    if ctx is None:
        ctx = _make_ctx(snapshot, player_id)

    hand = list(ctx.own_player.get("hand", []))
    vectors: List[float] = []
    for card in hand[:MAX_HAND_CARDS]:
        role = _role_from_card(card)
        card_id = str(card.get("card_id", ""))
        uid = str(card.get("uid", ""))
        card_actions = ctx.actions_by_source.get(uid, [])
        deployable = 1.0 if any(str(a.get("effect_id", "")) == "DeployUnit" for a in card_actions) else 0.0
        skill_usable = 1.0 if any(
            str(a.get("effect_id", "")) not in {"DeployUnit", "DefaultMove", "DefaultAttack", "TurnEnd"}
            for a in card_actions
        ) else 0.0
        vectors.extend([
            1.0,
            1.0 if card_id.startswith("Or_") else 0.0,
            1.0 if card_id.startswith("Cl_") else 0.0,
            deployable,
            skill_usable,
            *_role_one_hot(role),
        ])
    missing_slots = MAX_HAND_CARDS - min(len(hand), MAX_HAND_CARDS)
    if missing_slots > 0:
        vectors.extend([0.0] * missing_slots * 10)
    return vectors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_fixed_state_vector(snapshot: Dict[str, Any], player_id: Optional[str] = None) -> List[float]:
    player_id = player_id or snapshot.get("active_player", "P1")
    snapshot = _canonicalize_snapshot(snapshot, player_id)
    ctx = _make_ctx(snapshot, player_id)
    global_vector = _build_global_vector(snapshot, player_id, ctx)
    board_vector = _build_board_vector(snapshot, player_id, ctx)
    hand_vector = _build_hand_vector(snapshot, player_id, ctx)
    return global_vector + board_vector + hand_vector


def encode_action_features(
    snapshot: Dict[str, Any],
    action: Dict[str, Any],
    player_id: Optional[str] = None,
    _ctx: Optional[_Ctx] = None,
) -> List[float]:
    """액션 하나의 feature vector를 반환한다.

    _ctx가 제공되면 (build_observation 내부에서 호출 시) 재사용한다.
    외부에서 단독 호출할 경우 _ctx=None으로 남겨두면 내부에서 생성한다.
    """
    player_id = player_id or snapshot.get("active_player", "P1")
    if _ctx is None:
        snapshot = _canonicalize_snapshot(snapshot, player_id)
        action = _canonicalize_action(action, player_id)
        ctx = _make_ctx(snapshot, player_id)
    else:
        ctx = _ctx

    own_leader = ctx.own_leader
    enemy_leader = ctx.enemy_leader

    effect_id = str(action.get("effect_id", ""))
    target = action.get("target", {})
    target_type = str(target.get("type", "None"))

    # O(1) 조회 (이전에는 _find_card() O(N) 탐색)
    source = ctx.board_by_uid.get(str(action.get("source", "")))
    target_card = ctx.board_by_uid.get(str(target.get("guid", ""))) if target_type in {"Unit", "Card"} else None
    target_card2 = ctx.board_by_uid.get(str(target.get("guid2", ""))) if target_type == "Unit2" else None

    target_x = int(target.get("pos_x", -1))
    target_y = int(target.get("pos_y", -1))
    source_x = int(source.get("pos_x", -1)) if source else -1
    source_y = int(source.get("pos_y", -1)) if source else -1
    enemy_lx = int(enemy_leader.get("pos_x", -1)) if enemy_leader else -1
    enemy_ly = int(enemy_leader.get("pos_y", -1)) if enemy_leader else -1

    attack_mod, has_move_lock, has_attack_lock, timed_status_count = _status_summary(source) if source else (0.0, 0.0, 0.0, 0.0)
    source_role = _role_from_card(source or {})
    target_role = _role_from_card(target_card or {})

    # ctx.enemy_adj O(1) 조회 (이전에는 O(N) 보드 탐색)
    source_uid = str(source.get("uid", "")) if source else ""
    target_uid = str(target_card.get("uid", "")) if target_card else ""
    source_adjacent_enemies = ctx.enemy_adj.get(source_uid, 0.0) if source else 0.0
    target_incoming_attackers = ctx.enemy_adj.get(target_uid, 0.0) if target_card else 0.0

    move_distance_before = _distance(source_x, source_y, enemy_lx, enemy_ly)
    move_distance_after = _distance(target_x, target_y, enemy_lx, enemy_ly) if target_type == "Cell" else move_distance_before
    moves_closer = 1.0 if move_distance_after >= 0 and move_distance_before >= 0 and move_distance_after < move_distance_before else 0.0
    enters_leader_zone = 1.0 if target_type == "Cell" and move_distance_after >= 0 and move_distance_after <= (1.0 / 5.0) else 0.0

    target_hp = float(target_card.get("hp", 0.0)) if target_card else 0.0
    target_max_hp = max(1.0, float(target_card.get("max_hp", 1.0))) if target_card else 1.0
    source_atk = float(source.get("effective_atk", 0.0)) if source else 0.0
    can_kill_target = 1.0 if target_card and source_atk >= target_hp > 0 else 0.0
    threatens_enemy_leader = 1.0 if target_card and target_card.get("owner") != player_id and target_role == "Leader" else 0.0
    affects_two_units = 1.0 if target_card2 is not None else 0.0
    source_survives_trade = 1.0 if target_card and source is not None and float(target_card.get("effective_atk", 0.0)) < float(source.get("hp", 0.0)) else 0.0
    target_is_low_hp = 1.0 if target_card and target_hp <= 2.0 else 0.0
    source_from_hand = 1.0 if source is not None and not source.get("is_placed") else 0.0

    return [
        *_effect_one_hot(effect_id),
        *_target_one_hot(target_type),
        1.0 if effect_id == "TurnEnd" else 0.0,
        1.0 if effect_id == "DeployUnit" else 0.0,
        1.0 if effect_id == "DefaultMove" else 0.0,
        1.0 if effect_id == "DefaultAttack" else 0.0,
        0.0 if source is None else (1.0 if source.get("owner") == player_id else -1.0),
        0.0 if source is None else _normalize_ratio(float(source.get("atk", 0.0)), 10.0),
        0.0 if source is None else _normalize_ratio(source_atk, 10.0),
        0.0 if source is None else _normalize_ratio(float(source.get("hp", 0.0)), max(1.0, float(source.get("max_hp", 1.0)))),
        _normalize_ratio(attack_mod, 5.0),
        has_move_lock,
        has_attack_lock,
        _normalize_ratio(timed_status_count, 4.0),
        *_role_one_hot(source_role),
        0.0 if target_card is None else (1.0 if target_card.get("owner") != player_id else -1.0),
        0.0 if target_card is None else _normalize_ratio(float(target_card.get("effective_atk", 0.0)), 10.0),
        0.0 if target_card is None else _normalize_ratio(target_hp, target_max_hp),
        0.0 if target_card is None else (1.0 if target_role == "Leader" else 0.0),
        *_role_one_hot(target_role),
        _normalize_pos(source_x),
        _normalize_pos(source_y),
        _normalize_pos(target_x),
        _normalize_pos(target_y),
        move_distance_before if move_distance_before >= 0 else 0.0,
        move_distance_after if move_distance_after >= 0 else 0.0,
        moves_closer,
        enters_leader_zone,
        can_kill_target,
        threatens_enemy_leader,
        affects_two_units,
        source_survives_trade,
        target_is_low_hp,
        source_from_hand,
        _normalize_ratio(source_adjacent_enemies, 6.0),
        _normalize_ratio(target_incoming_attackers, 6.0),
    ]


ACTION_FEATURE_DIM = len(
    encode_action_features(
        {"board": [], "actions": [], "players": [], "active_player": "P1"},
        {"effect_id": "TurnEnd", "target": {"type": "None", "pos_x": -1, "pos_y": -1}},
    )
)


def build_observation(snapshot: Dict[str, Any], player_id: Optional[str] = None) -> SeaEngineObservation:
    player_id = player_id or snapshot.get("active_player", "P1")
    snapshot = _canonicalize_snapshot(snapshot, player_id)

    # _Ctx를 딱 한 번 만들고 모든 내부 함수에 재사용
    ctx = _make_ctx(snapshot, player_id)

    # _sorted_board_cards 인라인 (ctx.board 재사용)
    role_rank = {name: idx for idx, name in enumerate(ROLE_ORDER)}
    unit_list = sorted(
        ctx.board,
        key=lambda c: (c.get("owner", ""), role_rank.get(_role_from_card(c), 99), c.get("uid", "")),
    )
    hand_list = list(ctx.own_player.get("hand", []))

    global_vec = _build_global_vector(snapshot, player_id, ctx)
    board_vec = _build_board_vector(snapshot, player_id, ctx)
    hand_vec = _build_hand_vector(snapshot, player_id, ctx)
    state_vector = global_vec + board_vec + hand_vec

    # 각 액션 feature 계산 시 ctx 재사용 → _get_players/_get_leaders/_find_card 零 호출
    action_feature_vectors = [
        encode_action_features(snapshot, action, player_id, _ctx=ctx)
        for action in ctx.actions
    ]

    return SeaEngineObservation(
        unit_list=unit_list,
        hand_list=hand_list,
        global_vector=global_vec,
        legal_action_mask=[1 for _ in ctx.actions],
        state_vector=state_vector,
        action_feature_vectors=action_feature_vectors,
    )
