using System.Diagnostics;
using SeaEngine.Common;

namespace SeaEngine.CardManager;

public class CardLoader
{
    private static readonly CardData ErrorCard =
        new CardData("Er_L", "Error Card", "Er_L", UnitType.Leader, 1, 1);

    private readonly Dictionary<string, CardData> _cards = new Dictionary<string, CardData>();

    public CardLoader(string[] cardLines)
    {
        //ID,Name,LeaderID,UnitType,Atk,Hp,EffectID,EventID
        foreach (var line in cardLines.Skip(1))
        {
            string[] data = line.Split(',');
            if (data.Length < 6 || string.IsNullOrEmpty(data[0])) continue;

            UnitType unitType = data[3] switch
            {
                "L" => UnitType.Leader,
                "R" => UnitType.Rook,
                "P" => UnitType.Pawn,
                "B" => UnitType.Bishop,
                "N" => UnitType.Knight,
                _ => UnitType.Pawn
            };

            var id = data[0];
            var effId = data.Length >= 7 ? data[6] : null;
            var evtId = data.Length >= 8 ? data[7] : null;

            _cards.Add(id, new CardData(
                id,
                data[1],
                data[2],
                unitType, 
                int.Parse(data[4]),
                int.Parse(data[5]),
                string.IsNullOrEmpty(effId) ? null : effId,
                string.IsNullOrEmpty(evtId) ? null : evtId
            ));
            Console.WriteLine($"{id} loaded");
        }
    }

    public CardData GetCard(string cardName)
    {
        return _cards.GetValueOrDefault(cardName, ErrorCard);
    }
}
