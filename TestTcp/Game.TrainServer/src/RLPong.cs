using System.Text;
using Game.Network;
using SeaEngine.Common;

namespace Game.Server
{
    public class RLPong : INetReceiveEventHandler
    {
        public int HandlerId => RLConstant.HandlerId.RLPing;
        private INetAPI _net;
        private RLPeers _peers;

        public RLPong(INetAPI net, RLPeers peers)
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
                    _net.Send(RLConstant.HandlerId.RLRSP, queryNum, connId, Array.Empty<byte>());
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