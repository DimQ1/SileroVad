using Xunit;

namespace SileroVad.Tests
{
    public class VadStreamTests
    {
        public static TheoryData<VadModelKind> AllKinds => new() { VadModelKind.V4, VadModelKind.V5 };

        [Theory]
        [MemberData(nameof(AllKinds))]
        public void Streaming_MatchesBatch(VadModelKind kind)
        {
            using var vad = new Vad(kind);
            var expected = vad.GetSpeechTimestamps(TestAudio.Speech, new VadOptions());
            var actual = RunStream(vad, TestAudio.Speech, chunkSize: 1000);

            Assert.Equal(expected, actual);
        }

        [Theory]
        [MemberData(nameof(AllKinds))]
        public void Streaming_IsIndependentOfChunkSize(VadModelKind kind)
        {
            using var vad = new Vad(kind);

            var oneSampleAtATime = RunStream(vad, TestAudio.Speech, chunkSize: 1);
            var large = RunStream(vad, TestAudio.Speech, chunkSize: 4096);

            Assert.Equal(large, oneSampleAtATime);
        }

        [Theory]
        [MemberData(nameof(AllKinds))]
        public void Silence_ProducesNoSpeech(VadModelKind kind)
        {
            using var vad = new Vad(kind);

            Assert.Empty(RunStream(vad, TestAudio.Silence, chunkSize: 777));
        }

        [Fact]
        public void Reset_StartsANewStream()
        {
            using var vad = new Vad();
            using var stream = vad.CreateStream();

            var first = Push(stream, TestAudio.Speech, chunkSize: 500);
            stream.Reset();
            var second = Push(stream, TestAudio.Speech, chunkSize: 500);

            Assert.Equal(first, second);
            Assert.Equal(TestAudio.Speech.Length, stream.SamplesProcessed);
        }

        [Fact]
        public void Flush_WithoutAudio_ReturnsNothing()
        {
            using var vad = new Vad();
            using var stream = vad.CreateStream();

            Assert.Empty(stream.Flush());
            Assert.Equal(0, stream.SamplesProcessed);
        }

        [Fact]
        public void PartialWindow_IsProcessedOnFlush()
        {
            using var vad = new Vad();
            using var stream = vad.CreateStream();

            // Fewer samples than one window: nothing is complete until the stream is flushed.
            var pushed = stream.Push(TestAudio.Speech.AsSpan(0, 100));

            Assert.Empty(pushed);
            Assert.Equal(100, stream.SamplesProcessed);

            stream.Flush();

            Assert.Equal(100, stream.SamplesProcessed);
        }

        [Theory]
        [InlineData(16000, 1)]
        [InlineData(8000, 1)]
        [InlineData(48000, 3)]
        public void DecimationFactor_ReflectsSampleRate(int sampleRate, int expectedFactor)
        {
            using var vad = new Vad();
            using var stream = vad.CreateStream(new VadOptions { SampleRate = sampleRate });

            Assert.Equal(expectedFactor, stream.DecimationFactor);
            Assert.Equal(sampleRate, stream.SampleRate);
            Assert.Equal(sampleRate == 8000 ? 256 : 512, stream.WindowSizeSamples);
        }

        [Fact]
        public void HighSampleRate_ReportsPositionsInOriginalSamples()
        {
            using var vad = new Vad();
            var batch = vad.GetSpeechTimestamps(TestAudio.Speech, new VadOptions());

            var upsampled = TestAudio.Upsample(TestAudio.Speech, 3);
            using var stream = vad.CreateStream(new VadOptions { SampleRate = 48000 });
            var streamed = Collect(stream, upsampled, chunkSize: 3001);

            Assert.Equal(batch.Count, streamed.Count);

            for (var i = 0; i < batch.Count; i++)
            {
                Assert.Equal(batch[i].Start * 3, streamed[i].Start);
                Assert.Equal(batch[i].End * 3, streamed[i].End);
            }
        }

        [Fact]
        public void DisposedStream_Throws()
        {
            using var vad = new Vad();
            var stream = vad.CreateStream();
            stream.Dispose();

            Assert.Throws<ObjectDisposedException>(() => stream.Push(TestAudio.Silence));
        }

        private static List<VadSpeech> RunStream(Vad vad, float[] audio, int chunkSize) =>
            Collect(vad.CreateStream(), audio, chunkSize);

        /// <summary>Pushes audio into an existing stream without disposing it.</summary>
        private static List<VadSpeech> Push(VadStream stream, float[] audio, int chunkSize)
        {
            var segments = new List<VadSpeech>();

            for (var offset = 0; offset < audio.Length; offset += chunkSize)
            {
                var length = Math.Min(chunkSize, audio.Length - offset);
                segments.AddRange(stream.Push(audio.AsSpan(offset, length)));
            }

            segments.AddRange(stream.Flush());
            return segments;
        }

        private static List<VadSpeech> Collect(VadStream stream, float[] audio, int chunkSize)
        {
            using (stream)
            {
                return Push(stream, audio, chunkSize);
            }
        }
    }
}
