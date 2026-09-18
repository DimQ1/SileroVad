using SileroVad;
using SileroVad.Samples;

// Reads an audio file, prints the detected speech segments and writes them to '<input>.speech.wav'.
//
//   dotnet run --project SileroVad.Samples
//   dotnet run --project SileroVad.Samples -- "C:\path\to\audio.wav"

var filePath = args.Length > 0
    ? args[0]
    : Path.Combine(AppContext.BaseDirectory, "assets", "sample.wav");

if (!File.Exists(filePath))
{
    Console.Error.WriteLine($"File not found: '{filePath}'");
    Console.Error.WriteLine("Usage: dotnet run --project SileroVad.Samples -- <path to a wav file>");
    return 1;
}

Console.WriteLine($"Analysing '{Path.GetFullPath(filePath)}'");

var result = SpeechExtractor.ExtractSpeech(filePath);

Console.WriteLine($"Sample rate  : {result.SampleRate} Hz");
Console.WriteLine($"Speech parts : {result.Segments.Count}");

foreach (var segment in result.Segments)
{
    var start = segment.Start / (double)result.SampleRate;
    var end = segment.End / (double)result.SampleRate;
    Console.WriteLine($"  {start,7:F3}s .. {end,7:F3}s  ({segment.Duration(result.SampleRate).TotalMilliseconds,7:F0} ms)");
}

Console.WriteLine($"Total speech : {VadHelper.GetSpeechDuration(result.Segments, result.SampleRate).TotalSeconds:F2}s");
Console.WriteLine($"Written to   : {result.OutputPath}");

return 0;
