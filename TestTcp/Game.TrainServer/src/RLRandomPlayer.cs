using System.Runtime.InteropServices;
using System.Text;
using Game.Network;

namespace Game.Server
{
    public class RLRamdomPlayer : INetReceiveEventHandler
    {
        public int HandlerId => RLConstant.HandlerId.RLClose;
        private Random _rng = new();
        private INetAPI _net;
        private RLPeers _peers;
        

        public RLRamdomPlayer(INetAPI net, RLPeers peers)
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
                    while ( !engine.HasWinnder() && !engine.IsActivePlayer(learnerId)
                           && safetyLimit < 2000)
                    {
                        safetyLimit++;
                        var available = engine.game.Actions;
                        if (available.Count == 0) break;
                        var chosen = available[_rng.Next(available.Count)];
                        engine.UseAction(chosen.Guid, chosen.EffectId == "TurnEnd");
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
    }
}