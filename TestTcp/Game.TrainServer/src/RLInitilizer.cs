using Newtonsoft.Json;

using Game.Network;
using System.Text;
using System.Buffers.Binary;

namespace Game.Server
{
    public class RLInitilizer : INetReceiveEventHandler
    {
        public int HandlerId => RLConstant.HandlerId.RLInit;
        private INetAPI _net;
        private RLPeers _peers;

        public RLInitilizer(INetAPI net, RLPeers peers)
        {
            _net = net;
            _peers = peers;
        }
        // Data
        public void OnQuery(ConnId connId, int queryNum, byte[] raw)
        {
            if (!_peers.Rooms.ContainsKey(connId))
            {
                try
                {
                    var req = PayloadParser.ParseInit(raw);
                    var engine = new RLEngine(req);
                    _peers.Rooms.Add(connId, engine);

                    var payload = engine.SnapShot();
                    _net.Send(RLConstant.HandlerId.RLRSP, queryNum, connId, payload);
                }
                catch
                {
                    _net.Send(RLConstant.HandlerId.RLError, queryNum, connId, Array.Empty<byte>());
                } 
            }
            else _net.Send(RLConstant.HandlerId.RLError, queryNum, connId, Array.Empty<byte>());
        }
    }



    public static class PayloadParser
    {
        public static InitData ParseInit(byte[] payload)
            => JsonConvert.DeserializeObject<InitData>(Encoding.UTF8.GetString(payload));
    }

    public class InitData
    {
        [JsonProperty("command")]
        public string Command { get; set; }

        [JsonProperty("card_data_path")]
        public string CardDataPath { get; set; }

        [JsonProperty("player1_deck")]
        public string Player1Deck { get; set; }

        [JsonProperty("player2_deck")]
        public string Player2Deck { get; set; }

        [JsonProperty("player1_id")]
        public string Player1Id { get; set; }

        [JsonProperty("player2_id")]
        public string Player2Id { get; set; }
    }
}