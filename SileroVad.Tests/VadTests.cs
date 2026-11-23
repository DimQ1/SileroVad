using Xunit;

namespace SileroVad.Tests
{
    public class VadTests
    {
        public static TheoryData<VadModelKind> AllKinds => new() { VadModelKind.V4, VadModelKind.V5 };

        [Fact]
        public void DefaultConstructor_UsesLatestBundledModel()
        {
            using var vad = new Vad();

            Assert.Equal(VadModelKind.V5, vad.ModelKind);
        }

        [Theory]
        [MemberData(nameof(AllKinds))]
        public void BundledModels_AreLoaded(VadModelKind kind)
        {
            using var vad = new Vad(kind);

            Assert.Equal(kind, vad.ModelKind);
        }

        [Theory]
        [MemberData(nameof(AllKinds))]
        public void Silence_ProducesNoSpeech(VadModelKind kind)
        {
            using var vad = new Vad(kind);

            var segments = vad.GetSpeechTimestamps(TestAudio.Silence, new VadOptions());

            Assert.Empty(segments);
        }

        [Theory]
        [MemberData(nameof(AllKinds))]
        public void Speech_IsDetected(VadModelKind kind)
        {
            using var vad = new Vad(kind);

            var segments = vad.GetSpeechTimestamps(TestAudio.Speech, new VadOptions());

            Assert.NotEmpty(segments);

            // The excerpt is continuous speech, so almost all of it must be reported.
            var speechSamples = segments.Sum(s => s.Length);
            Assert.True(
                speechSamples > TestAudio.Speech.Length * 0.9,
                $"{kind}: expected most of the excerpt to be speech, got {speechSamples} of {TestAudio.Speech.Length} samples.");

            Assert.All(segments, s =>
            {
                Assert.InRange(s.Start, 0, TestAudio.Speech.Length);
                Assert.InRange(s.End, 0, TestAudio.Speech.Length);
                Assert.True(s.End > s.Start);
            });
        }

        [Theory]
        [MemberData(nameof(AllKinds))]
        public void LegacyOverload_MatchesOptionsOverload(VadModelKind kind)
        {
            using var vad = new Vad(kind);

            var legacy = vad.GetSpeechTimestamps(TestAudio.Speech);

            // The historical overload always uses 16 kHz and a 1024 sample window; v5 models only accept 512.
            var expected = vad.GetSpeechTimestamps(
                TestAudio.Speech,
                new VadOptions { WindowSizeSamples = kind == VadModelKind.V4 ? 1024 : 512 });

            Assert.Equal(expected, legacy);
        }

        [Theory]
        [MemberData(nameof(AllKinds))]
        public void SpeechProbabilities_AreInRange(VadModelKind kind)
        {
            using var vad = new Vad(kind);

            var probabilities = vad.GetSpeechProbabilities(TestAudio.Speech);

            var windowSize = kind == VadModelKind.V5 ? 512 : 1024;
            Assert.Equal((TestAudio.Speech.Length + windowSize - 1) / windowSize, probabilities.Length);
            Assert.All(probabilities, p => Assert.InRange(p, 0f, 1f));
            Assert.Contains(probabilities, p => p >= 0.5f);
        }

        [Theory]
        [MemberData(nameof(AllKinds))]
        public void SpeechProbabilities_RespectWindowSizeOption(VadModelKind kind)
        {
            using var vad = new Vad(kind);

            // 512 samples is a supported window for both revisions.
            const int windowSize = 512;
            var probabilities = vad.GetSpeechProbabilities(
                TestAudio.Speech,
                new VadOptions { WindowSizeSamples = windowSize });

            Assert.Equal((TestAudio.Speech.Length + windowSize - 1) / windowSize, probabilities.Length);
        }

        [Fact]
        public void V5Model_IgnoresWindowSizeOptionItCannotAccept()
        {
            using var vad = new Vad(VadModelKind.V5);

            var probabilities = vad.GetSpeechProbabilities(
                TestAudio.Speech,
                new VadOptions { WindowSizeSamples = 256 });

            // v5 requires 512 samples per call at 16 kHz, so the unsupported option is ignored.
            Assert.Equal((TestAudio.Speech.Length + 511) / 512, probabilities.Length);
        }

        [Theory]
        [MemberData(nameof(AllKinds))]
        public void EightKhz_AgreesWithSixteenKhz(VadModelKind kind)
        {
            using var vad = new Vad(kind);

            var at16k = vad.GetSpeechTimestamps(TestAudio.Speech, new VadOptions());
            var at8k = vad.GetSpeechTimestamps(
                TestAudio.Decimate(TestAudio.Speech, 2),
                new VadOptions { SampleRate = 8000 });

            Assert.Equal(at16k.Count, at8k.Count);

            var speech16k = at16k.Sum(s => s.Length) / 16000.0;
            var speech8k = at8k.Sum(s => s.Length) / 8000.0;

            Assert.True(
                Math.Abs(speech16k - speech8k) < 0.25,
                $"{kind}: speech duration differs between rates: {speech16k:F2}s at 16 kHz vs {speech8k:F2}s at 8 kHz.");
        }

        [Theory]
        [MemberData(nameof(AllKinds))]
        public void FortyEightKhz_ReportsPositionsInOriginalSamples(VadModelKind kind)
        {
            using var vad = new Vad(kind);

            var at16k = vad.GetSpeechTimestamps(TestAudio.Speech, new VadOptions());
            var at48k = vad.GetSpeechTimestamps(
                TestAudio.Upsample(TestAudio.Speech, 3),
                new VadOptions { SampleRate = 48000 });

            // Audio is decimated internally by 3 and reported positions are scaled back.
            Assert.Equal(at16k.Count, at48k.Count);

            for (var i = 0; i < at16k.Count; i++)
            {
                Assert.Equal(at16k[i].Start * 3, at48k[i].Start);
                Assert.Equal(at16k[i].End * 3, at48k[i].End);
            }
        }

        [Fact]
        public void UnsupportedSampleRate_Throws()
        {
            using var vad = new Vad();

            Assert.Throws<ArgumentOutOfRangeException>(
                () => vad.GetSpeechTimestamps(TestAudio.Silence, new VadOptions { SampleRate = 11025 }));
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(-0.5f)]
        [InlineData(1.5f)]
        public void InvalidThreshold_Throws(float threshold)
        {
            using var vad = new Vad();

            Assert.Throws<ArgumentOutOfRangeException>(
                () => vad.GetSpeechTimestamps(TestAudio.Silence, new VadOptions { Threshold = threshold }));
        }

        [Fact]
        public void EmptyAudio_ProducesNoSpeech()
        {
            using var vad = new Vad();

            var segments = vad.GetSpeechTimestamps(Array.Empty<float>(), new VadOptions());

            Assert.Empty(segments);
        }

        [Fact]
        public void PathConstructor_DetectsModelRevision()
        {
            using var v5 = new Vad(TestAudio.V5ModelPath);
            using var v4 = new Vad(TestAudio.V4ModelPath);

            Assert.Equal(VadModelKind.V5, v5.ModelKind);
            Assert.Equal(VadModelKind.V4, v4.ModelKind);
        }

        [Fact]
        public void SharedModel_IsNotDisposedByDetector()
        {
            using var model = new VadModel(TestAudio.V5ModelPath);

            using (var vad = new Vad(model))
            {
                Assert.Equal(VadModelKind.V5, vad.ModelKind);
            }

            // The detector must not dispose a model it did not create.
            var probability = model.DetectSpeech(new float[512], model.CreateState(), TestAudio.SampleRate);

            Assert.InRange(probability, 0f, 1f);
        }

        [Fact]
        public void UnknownBundledModel_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new Vad((VadModelKind)3));
        }

        [Theory]
        [MemberData(nameof(AllKinds))]
        public void BytesConstructor_MatchesBundledModel(VadModelKind kind)
        {
            var path = kind == VadModelKind.V5 ? TestAudio.V5ModelPath : TestAudio.V4ModelPath;

            using var fromBytes = new Vad(File.ReadAllBytes(path));
            using var bundled = new Vad(kind);

            Assert.Equal(kind, fromBytes.ModelKind);
            Assert.Equal(
                bundled.GetSpeechTimestamps(TestAudio.Speech, new VadOptions()),
                fromBytes.GetSpeechTimestamps(TestAudio.Speech, new VadOptions()));
        }

        [Fact]
        public void LegacyOverload_AcceptsAnUnusualWindowForV4()
        {
            // 768 samples is not one of the trained window sizes: a warning is printed, detection still runs.
            using var vad = new Vad(VadModelKind.V4);

            Assert.NotEmpty(vad.GetSpeechTimestamps(TestAudio.Speech, window_size_samples: 768));
        }

        [Theory]
        [InlineData(nameof(VadOptions.Threshold))]
        [InlineData(nameof(VadOptions.MinSpeechDurationMs))]
        [InlineData(nameof(VadOptions.MinSilenceDurationMs))]
        [InlineData(nameof(VadOptions.MinSilenceAtMaxSpeechMs))]
        [InlineData(nameof(VadOptions.SpeechPadMs))]
        [InlineData(nameof(VadOptions.WindowSizeSamples))]
        [InlineData(nameof(VadOptions.MaxSpeechDurationSeconds))]
        [InlineData(nameof(VadOptions.NegThreshold))]
        public void InvalidOption_IsRejectedEverywhere(string option)
        {
            var options = option switch
            {
                nameof(VadOptions.Threshold) => new VadOptions { Threshold = 1.5f },
                nameof(VadOptions.MinSpeechDurationMs) => new VadOptions { MinSpeechDurationMs = -1 },
                nameof(VadOptions.MinSilenceDurationMs) => new VadOptions { MinSilenceDurationMs = -1 },
                nameof(VadOptions.MinSilenceAtMaxSpeechMs) => new VadOptions { MinSilenceAtMaxSpeechMs = -1 },
                nameof(VadOptions.SpeechPadMs) => new VadOptions { SpeechPadMs = -1 },
                nameof(VadOptions.WindowSizeSamples) => new VadOptions { WindowSizeSamples = 0 },
                nameof(VadOptions.MaxSpeechDurationSeconds) => new VadOptions { MaxSpeechDurationSeconds = 0f },
                _ => new VadOptions { NegThreshold = -0.1f },
            };

            using var vad = new Vad();

            Assert.Equal(option, Assert.Throws<ArgumentOutOfRangeException>(() => vad.GetSpeechTimestamps(TestAudio.Silence, options)).ParamName);
            Assert.Equal(option, Assert.Throws<ArgumentOutOfRangeException>(() => vad.GetSpeechProbabilities(TestAudio.Silence, options)).ParamName);
            Assert.Equal(option, Assert.Throws<ArgumentOutOfRangeException>(() => vad.GetSpeechTimestampsFromProbabilities(new[] { 1f }, options)).ParamName);
            Assert.Equal(option, Assert.Throws<ArgumentOutOfRangeException>(() => vad.CreateStream(options)).ParamName);
        }

        [Fact]
        public void NullOptions_Throw()
        {
            using var vad = new Vad();

            Assert.Throws<ArgumentNullException>(() => vad.GetSpeechTimestamps(TestAudio.Silence, (VadOptions)null!));
            Assert.Throws<ArgumentNullException>(() => vad.GetSpeechTimestampsFromProbabilities(new[] { 1f }, null!));
        }
    }
}
