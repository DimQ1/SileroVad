namespace SileroVad.Tests
{
    /// <summary>
    /// Audio used by the tests.
    /// </summary>
    internal static class TestAudio
    {
        internal const int SampleRate = 16000;

        /// <summary>
        /// Four seconds of continuous speech at 16 kHz, taken from <c>tests/data/test.wav</c> of the
        /// reference implementation (<see href="https://github.com/snakers4/silero-vad"/>).
        /// </summary>
        internal static float[] Speech { get; } = ReadWav(Path.Combine(AppContext.BaseDirectory, "assets", "speech-16k.wav"));

        /// <summary>Three seconds of digital silence.</summary>
        internal static float[] Silence { get; } = new float[SampleRate * 3];

        internal static string ModelPath(string fileName) =>
            Path.Combine(AppContext.BaseDirectory, "models", fileName);

        internal static string V5ModelPath => ModelPath("silero_vad.onnx");

        internal static string V4ModelPath => ModelPath("silero_vad_v4.onnx");

        /// <summary>Decimates audio by an integer factor, the way the reference implementation does.</summary>
        internal static float[] Decimate(float[] audio, int factor)
        {
            var result = new float[audio.Length / factor];
            for (var i = 0; i < result.Length; i++)
            {
                result[i] = audio[i * factor];
            }

            return result;
        }

        /// <summary>Repeats every sample <paramref name="factor"/> times, producing audio at a higher rate.</summary>
        internal static float[] Upsample(float[] audio, int factor)
        {
            var result = new float[audio.Length * factor];
            for (var i = 0; i < audio.Length; i++)
            {
                for (var j = 0; j < factor; j++)
                {
                    result[(i * factor) + j] = audio[i];
                }
            }

            return result;
        }

        private static float[] ReadWav(string path)
        {
            var bytes = File.ReadAllBytes(path);
            var position = 12;

            while (position + 8 <= bytes.Length)
            {
                var id = System.Text.Encoding.ASCII.GetString(bytes, position, 4);
                var size = BitConverter.ToInt32(bytes, position + 4);

                if (id == "data")
                {
                    var samples = new float[size / 2];
                    for (var i = 0; i < samples.Length; i++)
                    {
                        samples[i] = BitConverter.ToInt16(bytes, position + 8 + (i * 2)) / 32768f;
                    }

                    return samples;
                }

                position += 8 + size + (size % 2);
            }

            throw new InvalidDataException($"No data chunk found in '{path}'.");
        }
    }
}
