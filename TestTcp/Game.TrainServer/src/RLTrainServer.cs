using System.ComponentModel;
using Game.Network;

namespace Game.Server
{
    public class RLPeers
    {
        public Dictionary<ConnId, RLEngine> Rooms = new();
    }
    public class RLTrainServer
    {
        private RLPeers _peers = new();
        private NetworkManager _net;
        private RLControl _control;
        private RLInitilizer _entrance;
        private RLApplier _applier;
        private RLPong _pong;
        private RLCloser _closer;
        private RLGreedyPlayer _greedy;
        private RLRamdomPlayer _random;
        private RLSnapShot _snapShot;

        public RLTrainServer()
        {
            _net = NetworkManager.CreateNetworkManager(9000, 30);
            _control = new(_net, _peers);
            _entrance = new(_net, _peers);
            _applier = new(_net, _peers);
            _pong = new(_net, _peers);
            _closer = new(_net, _peers);
            _greedy = new(_net, _peers);
            _random = new(_net, _peers);
            _snapShot = new(_net, _peers);

            _net.SetControlHandler(_control);
            _net.SetReceiveHandler(_entrance);
            _net.SetReceiveHandler(_applier);
            _net.SetReceiveHandler(_pong);
            _net.SetReceiveHandler(_closer);
            _net.SetReceiveHandler(_greedy);
            _net.SetReceiveHandler(_random);
            _net.SetReceiveHandler(_snapShot);
            

            _net.Start();
        }

        public void Tick()
        {
            _net.Tick();
        }

        public async Task End()
        {
            await _net.StopAsync();
        }
    }
}