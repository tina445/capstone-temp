
using System.Text;
using Game.Network;
using SeaEngine.Common;

namespace Game.Server
{
    public class RLGreedyPlayer : INetReceiveEventHandler
    {
        public int HandlerId => RLConstant.HandlerId.RLGreedyPlay;
        private Random _rng = new();
        private INetAPI _net;
        private RLPeers _peers;


        public RLGreedyPlayer(INetAPI net, RLPeers peers)
        {
            _net = net;
            _peers = peers;
        }
        // Data
        public void OnQuery(ConnId connId, int queryNum, byte[] raw)
        {
            if (_peers.Rooms.TryGetValue(connId, out var engine))
            {
                try
                {
                    var learnerId = Encoding.UTF8.GetString(raw);
                    int safetyLimit = 0;
                    while (!engine.HasWinnder() && !engine.IsActivePlayer(learnerId)
                           && safetyLimit < 2000)
                    {
                        safetyLimit++;
                        var available = engine.game.Actions;
                        if (available.Count == 0) break;
                        var action = ChooseGreedyAction(engine);
                        engine.UseAction(action.Guid, action.EffectId == "TurnEnd");
                    }

                    _net.Send(RLConstant.HandlerId.RLRSP, queryNum, connId, engine.SnapShot());
                }
                catch
                {
                    _net.Send(RLConstant.HandlerId.RLError, queryNum, connId, Array.Empty<byte>());
                }
            }
            else _net.Send(RLConstant.HandlerId.RLError, queryNum, connId, Array.Empty<byte>());
        }
        public GameAction ChooseGreedyAction(RLEngine engine)
        {
            // Uid.ToString() 기반으로 card 조회 (C# Uid는 System.Guid가 아님)
            var cardByGuidStr = engine.game.Data.Board.Cards
                .Where(c => c.Unit.IsPlaced)
                .ToDictionary(c => c.Guid.ToString());

            int Score(GameAction a)
            {
                var score = a.EffectId switch
                {
                    "DefaultAttack" => 80,
                    "DeployUnit" => 45,
                    "DefaultMove" => 20,
                    "TurnEnd" => -100,
                    _ => 55,   // 스킬
                };
                // 유닛 타깃이면 Leader 보너스 + 저체력 보너스 추가
                if (cardByGuidStr.TryGetValue(a.Target.Guid.ToString(), out var tCard))
                {
                    if (tCard.Data.UnitType.ToString() == "Leader") score += 100;
                    score += Math.Max(0, 10 - tCard.Unit.Hp);
                }
                return score + _rng.Next(4);  // 타이 브레이킹
            }

            return engine.game.Actions.MaxBy(Score)!;
        }
    }
}