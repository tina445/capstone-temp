using Game.Network;

namespace Game.Server
{
    public class RLControl : INetControlEventHandler
    {
        private INetAPI _net;
        private RLPeers _peers;

        public RLControl(INetAPI net, RLPeers peers)
        {
            _net = net;
            _peers = peers;
        }
        // Data
        public void OnDisconnect(ConnId connId, byte[] raw)
        {
            if (_peers.Rooms.Remove(connId, out var engine))
            {
                
            }
        }
    }
}