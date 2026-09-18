using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace SileroVad.Samples
{
    /// <summary>
    /// Result of <see cref="SpeechExtractor.ExtractSpeech"/>.
    /// </summary>
    /// <param name="OutputPath">Wav file the speech was written to.</param>
    /// <param name="Segments">Speech segments, in samples of <paramref name="SampleRate"/>.</param>
    /// <param name="SampleRate">Sample rate the audio was analysed at.</param>
    public sealed record SpeechExtractionResult(string OutputPath, IReadOnlyList<VadSpeech> Segments, int SampleRate);

    /// <summary>
    /// Reads an audio file, keeps the parts that silero VAD classifies as speech and writes them to
    /// <c>&lt;input&gt;.speech.wav</c> next to the input file.
    /// </summary>
    public static class SpeechExtractor
    {
        /// <summary>Rate used when the input has to be resampled.</summary>
        private const int FallbackSampleRate = 16000;

        /// <summary>One detector for the whole application; it is stateless between calls.</summary>
        private static readonly Vad Vad = new();

        /// <param name="filePath">Path to a <c>.wav</c> file.</param>
        /// <param name="options">
        /// Detection options; <see cref="VadOptions.SampleRate"/> is filled in automatically.
        /// </param>
        public static SpeechExtractionResult ExtractSpeech(string filePath, VadOptions? options = null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
            options ??= new VadOptions { MinSilenceDurationMs = 500, Threshold = 0.5f };

            var extension = Path.GetExtension(filePath).ToLowerInvariant();
            using WaveStream waveFileReader = extension switch
            {
                ".wav" => new WaveFileReader(filePath),
                // ".mp3" => new Mp3FileReader(filePath),
                //   NAudio 3.x moved Mp3FileReader into NAudio.WinMM, so that branch needs a Windows target
                //   framework (net10.0-windows). Decode mp3 with your own decoder or convert it first.
                _ => throw new NotSupportedException($"not supported {extension}"),
            };

            var sampleRate = waveFileReader.WaveFormat.SampleRate;
            ISampleProvider sampleProvider = waveFileReader.ToSampleProvider().ToMono();

            // Silero VAD works at 8000, 16000 and multiples of 16000 Hz, so resample anything else.
            if (!VadModel.IsSupportedSampleRate(sampleRate))
            {
                sampleProvider = new WdlResamplingSampleProvider(sampleProvider, FallbackSampleRate);
                sampleRate = FallbackSampleRate;
            }

            var samples = new float[(int)(waveFileReader.TotalTime.TotalSeconds * sampleRate)];
            var count = 0;
            int read;
            while (count < samples.Length && (read = sampleProvider.Read(samples.AsSpan(count))) > 0)
            {
                count += read;
            }

            var audio = samples[..count];

            var speeches = Vad.GetSpeechTimestamps(audio, options with { SampleRate = sampleRate });

            var audioSpeech = VadHelper.GetSpeechSamples(audio, speeches);

            var outputPath = Path.ChangeExtension(filePath, "speech") + ".wav";
            using var fileWriter = new WaveFileWriter(outputPath, new WaveFormat(sampleRate, 1));
            foreach (var sample in audioSpeech)
            {
                fileWriter.WriteSample(sample);
            }

            fileWriter.Flush();

            return new SpeechExtractionResult(outputPath, speeches, sampleRate);
        }
    }
}
