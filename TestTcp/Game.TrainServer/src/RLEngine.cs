
using System.Text;
using SeaEngine.Common;
using SeaEngine.Logger;

using System.Text.Json;
using System.Text.Json.Serialization;


namespace Game.Server
{
    public class RLEngine
    {
        private SeaEngine.Game _engine;
        private JsonSerializerOptions jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };

        private int _turnCounter = 1;


        public RLEngine(InitData initData)
        {
            _engine = new(new SeaEngine.CardManager.CardLoader(File.ReadAllLines(Setting.DBPath)), new SimpleLogger(), initData.Player1Id, initData.Player2Id);

            _engine.Init(initData.Player1Deck, initData.Player2Deck);
        }

        public bool HasWinnder() => _engine.Data.Winner != null;

        public bool IsActivePlayer(string learnerId) => _engine.Data.ActivePlayerId == learnerId;

        public SeaEngine.Game game => _engine;

        public void UseAction(Uid uid, bool turnEnd = false)
        {   
            _engine.UseAction(uid);
            if (turnEnd) _turnCounter++;
        }


        public byte[] SnapShot()
            => Encoding.UTF8.GetBytes(_engine.Serialize());

    };
}

