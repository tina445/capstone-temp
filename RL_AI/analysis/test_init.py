import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from SeaEngine.bridge.seaengine_session import SeaEngineSession

def main():
    session = SeaEngineSession()
    session.start()
    
    try:
        # Default initialization (as in trainer.py)
        snap = session.init_game(player1_deck="", player2_deck="")
        
        p1_cards = [c.get("name", c.get("card_id")) for c in snap.get("board", []) if c.get("owner") == "P1"]
        p2_cards = [c.get("name", c.get("card_id")) for c in snap.get("board", []) if c.get("owner") == "P2"]
        
        print("Default P1 Deck (in training):", set(p1_cards))
        print("Default P2 Deck (in training):", set(p2_cards))
        
    finally:
        session.close()

if __name__ == "__main__":
    main()