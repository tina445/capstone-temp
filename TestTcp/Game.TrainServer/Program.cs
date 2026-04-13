
using System.Diagnostics;
using Game.Server;
using Microsoft.VisualBasic;

namespace Game.Network;

class Program
{
    public const int TickTime = 15;
    static async Task Main()
    {
        Log.SetLogger(Console.WriteLine);

        RLTrainServer rLTrainServer = new();

        var stopwatch = new Stopwatch();
        long delta = 0;

        var cts = new CancellationTokenSource();

        var inputTask = Task.Run(() =>
        {
            while (true)
            {
                var line = Console.ReadLine();
                if (line != null && line.Trim().Equals("q", StringComparison.OrdinalIgnoreCase))
                {
                    cts.Cancel();
                    break;
                }
            }
        });

        try
        {
            while (!cts.IsCancellationRequested)
            {
                stopwatch.Restart();

                rLTrainServer.Tick();

                stopwatch.Stop();

                delta = stopwatch.ElapsedMilliseconds;
                int sleepTime = TickTime - (int)delta;
                if (sleepTime > 0) Thread.Sleep(sleepTime);

                else Log.WriteLog("TickTime over");
            }

        }
        catch (OperationCanceledException)
        {

        }
        finally
        {
            await rLTrainServer.End();
            Console.WriteLine("Server stopped.");
        }

        await rLTrainServer.End();
    }


}