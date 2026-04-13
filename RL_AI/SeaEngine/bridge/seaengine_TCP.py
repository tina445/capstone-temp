"""Bridge for running the copied SeaEngine C# project as the actual game backend."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class SeaEngineSessionTCP:
    def __init__(
        self,
        *,
        card_data_path: Optional[str] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parent.parent)
        self.card_data_path = str(
            Path(card_data_path).resolve()
            if card_data_path is not None
            else (self.project_root.parent / "cards" / "Cards.csv").resolve()
        )

        self._sock: Optional[socket.socket] = None
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None

        self._snapshot: Optional[Dict[str, Any]] = None

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        self._next_query_id = 1
        self._pending: Dict[int, Dict[str, Any]] = {}

    # =========================
    def start(self) -> None:
        if self._sock is not None:
            return

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect(("127.0.0.1", 9000))

        time.sleep(1)

        # 기존 PeerEntrance (이름 등록)
        self._send_packet(
            flag=0x00000000,
            handler=7,
            query_num=0,
            payload="AI Player".encode("utf-8"),
        )

        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    # =========================
    def close(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            finally:
                self._sock = None

    # =========================
    def ping(self) -> Dict[str, Any]:
        return {"status": "ok"}

    # =========================
    # 🔥 변경된 init_game
    # =========================
    def init_game(
        self,
        *,
        player1_deck: str = "",
        player2_deck: str = "",
        player1_id: str = "P1",
        player2_id: str = "P2",
    ) -> Dict[str, Any]:

        payload_dict = {
            "command": "init",
            "card_data_path": self.card_data_path,
            "player1_deck": player1_deck,
            "player2_deck": player2_deck,
            "player1_id": player1_id,
            "player2_id": player2_id,
        }

        payload_bytes = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")

        with self._cv:
            qid = self._next_query_id
            self._next_query_id += 1

            self._pending[qid] = {"done": False}

        # 🔥 PeerEntrance handler로 Query 전송
        self._send_packet(
            flag=0x00000004,  # Query
            handler=7,        # PeerEntrance
            query_num=qid,
            payload=payload_bytes,
        )

        # 🔥 응답 기다림 (내용 무시)
        with self._cv:
            while not self._pending[qid]["done"]:
                self._cv.wait()

            del self._pending[qid]

        return {"status": "ok"}

    # =========================
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._snapshot or {}

    # =========================
    def apply_action(self, action_uid: str) -> Dict[str, Any]:
        with self._cv:
            qid = self._next_query_id
            self._next_query_id += 1

            self._pending[qid] = {"done": False, "response": None}

        self._send_packet(
            flag=0x00000004,
            handler=6,
            query_num=qid,
            payload=action_uid.encode("utf-8"),
        )

        with self._cv:
            while not self._pending[qid]["done"]:
                self._cv.wait()

            response = self._pending[qid]["response"]
            del self._pending[qid]

        return response

    # =========================
    def _reader_loop(self):
        while self._running:
            try:
                pkt = self._read_packet()

                flag = pkt["flag"]
                handler = pkt["handler"]
                qid = pkt["query_num"]
                payload = pkt["payload"]

                data = None
                if payload:
                    try:
                        data = json.loads(payload.decode("utf-8"))
                    except:
                        data = None

                with self._cv:
                    # snapshot 갱신 (GameMessage만)
                    if handler == 6 and data is not None:
                        self._snapshot = data

                    # 🔥 모든 Respond 처리 (핸들러 무관)
                    if flag & 0x00000002:
                        if qid in self._pending:
                            self._pending[qid]["done"] = True
                            self._pending[qid]["response"] = data
                            self._cv.notify_all()

            except Exception as e:
                print("[ERROR]", e)
                self._running = False

    # =========================
    def _send_packet(self, flag, handler, query_num, payload):
        total_size = 16 + len(payload)

        packet = struct.pack(
            "<I I i i i",
            total_size,
            flag,
            handler,
            query_num,
            0,
        ) + payload

        self._sock.sendall(packet)

    # =========================
    def _read_exact(self, size):
        data = b""
        while len(data) < size:
            chunk = self._sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("socket closed")
            data += chunk
        return data

    def _read_packet(self):
        size_bytes = self._read_exact(4)
        (total_size,) = struct.unpack("<I", size_bytes)

        body = self._read_exact(total_size)

        header = body[:16]
        payload = body[16:]

        flag, handler, query_num, _ = struct.unpack("<I i i i", header)

        return {
            "flag": flag,
            "handler": handler,
            "query_num": query_num,
            "payload": payload,
        }