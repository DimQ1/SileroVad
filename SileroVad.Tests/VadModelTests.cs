using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace SileroVad.Tests
{
    public class VadModelTests
    {
        [Fact]
        public void Revision_IsDetectedFromTheModel()
        {
            using var v5 = new VadModel(TestAudio.V5ModelPath);
            using var v4 = new VadModel(TestAudio.V4ModelPath);

            Assert.Equal(VadModelKind.V5, v5.Kind);
            Assert.Equal(128, v5.StateSize);
            Assert.True(v5.IsWindowSizeFixed);
            Assert.Equal(512, v5.GetWindowSamples(16000));
            Assert.Equal(256, v5.GetWindowSamples(8000));
            Assert.Equal(64, v5.GetContextSamples(16000));
            Assert.Equal(32, v5.GetContextSamples(8000));

            Assert.Equal(VadModelKind.V4, v4.Kind);
            Assert.Equal(64, v4.StateSize);
            Assert.False(v4.IsWindowSizeFixed);
            Assert.Equal(1024, v4.GetWindowSamples(16000));
            Assert.Equal(0, v4.GetContextSamples(16000));
        }

        [Fact]
        public void Revision_IsDetectedFromBytes()
        {
            using var v5 = new VadModel(File.ReadAllBytes(TestAudio.V5ModelPath));
            using var v4 = new VadModel(File.ReadAllBytes(TestAudio.V4ModelPath));

            Assert.Equal(VadModelKind.V5, v5.Kind);
            Assert.Equal(VadModelKind.V4, v4.Kind);
        }

        [Theory]
        [InlineData(8000, true)]
        [InlineData(16000, true)]
        [InlineData(32000, true)]
        [InlineData(44100, false)]
        [InlineData(48000, true)]
        [InlineData(4000, false)]
        public void SupportedSampleRates(int sampleRate, bool expected)
        {
            Assert.Equal(expected, VadModel.IsSupportedSampleRate(sampleRate));
        }

        [Fact]
        public void DetectSpeech_ReturnsProbabilityInRange()
        {
            using var model = new VadModel(TestAudio.V5ModelPath);
            var state = model.CreateState(TestAudio.SampleRate);
            var window = new float[512];

            var probability = model.DetectSpeech(window, state, TestAudio.SampleRate);

            Assert.InRange(probability, 0f, 1f);
        }

        [Fact]
        public void DetectSpeech_AdvancesTheState()
        {
            using var model = new VadModel(TestAudio.V5ModelPath);
            var window = TestAudio.Speech.AsSpan(0, 512);

            var first = model.CreateState(TestAudio.SampleRate);
            var second = model.CreateState(TestAudio.SampleRate);

            var firstProbability = model.DetectSpeech(window, first, TestAudio.SampleRate);
            model.DetectSpeech(window, second, TestAudio.SampleRate);
            var repeated = model.DetectSpeech(window, first, TestAudio.SampleRate);

            // A stateful model gives a different answer for the same window once its state moved on.
            Assert.NotEqual(firstProbability, repeated);
        }

        [Fact]
        public void DetectSpeech_RejectsWrongWindowForV5()
        {
            using var model = new VadModel(TestAudio.V5ModelPath);
            var state = model.CreateState(TestAudio.SampleRate);

            var exception = Assert.Throws<ArgumentException>(
                () => model.DetectSpeech(new float[256], state, TestAudio.SampleRate));

            Assert.Contains("512", exception.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void DetectSpeech_RejectsMismatchedState()
        {
            using var model = new VadModel(TestAudio.V5ModelPath);
            using var v4Model = new VadModel(TestAudio.V4ModelPath);
            var state = v4Model.CreateState(TestAudio.SampleRate);

            Assert.Throws<ArgumentException>(() => model.DetectSpeech(new float[512], state, TestAudio.SampleRate));
        }

        [Fact]
        public void EffectiveSampleRate_RejectsUnsupportedRate()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => VadModel.EffectiveSampleRate(44100));
        }

        [Fact]
        public void Constructors_RejectMissingModel()
        {
            Assert.Throws<ArgumentNullException>(() => new VadModel((byte[])null!));
            Assert.Throws<ArgumentException>(() => new VadModel((string)null!));
            Assert.Throws<ArgumentException>(() => new VadModel("  "));
        }

        [Fact]
        public void DetectSpeech_RejectsStateOfAnotherSampleRate()
        {
            using var model = new VadModel(TestAudio.V5ModelPath);
            var state = model.CreateState(8000);

            var exception = Assert.Throws<ArgumentException>(
                () => model.DetectSpeech(new float[512], state, TestAudio.SampleRate));

            Assert.Contains("8000", exception.Message, StringComparison.Ordinal);
        }

        [Theory]
        [InlineData(VadModelKind.V4)]
        [InlineData(VadModelKind.V5)]
        public void DetectSpeech_RejectsEmptyWindow(VadModelKind kind)
        {
            using var model = new VadModel(kind == VadModelKind.V5 ? TestAudio.V5ModelPath : TestAudio.V4ModelPath);

            Assert.Throws<ArgumentException>(
                () => model.DetectSpeech(ReadOnlySpan<float>.Empty, model.CreateState(), TestAudio.SampleRate));
        }

        [Fact]
        public void CreateState_DescribesTheState()
        {
            using var v5 = new VadModel(TestAudio.V5ModelPath);
            using var v4 = new VadModel(TestAudio.V4ModelPath);

            var narrowband = v5.CreateState(8000, batchSize: 3);
            Assert.Equal(VadModelKind.V5, narrowband.Kind);
            Assert.Equal(8000, narrowband.SampleRate);
            Assert.Equal(3, narrowband.BatchSize);
            Assert.Equal(128, narrowband.StateSize);
            Assert.Equal(32, narrowband.ContextSamples);

            // Rates above 16 kHz are decimated, so the state works at 16 kHz.
            Assert.Equal(16000, v5.CreateState(48000).SampleRate);

            var legacy = v4.CreateState();
            Assert.Equal(VadModelKind.V4, legacy.Kind);
            Assert.Equal(64, legacy.StateSize);
            Assert.Equal(0, legacy.ContextSamples);
        }

        [Fact]
        public void StateReset_ClearsBothV4Tensors()
        {
            using var model = new VadModel(TestAudio.V4ModelPath);
            var window = TestAudio.Speech.AsSpan(0, 1024);

            var state = model.CreateState(TestAudio.SampleRate);
            var fresh = model.CreateState(TestAudio.SampleRate);

            model.DetectSpeech(window, state, TestAudio.SampleRate);
            model.DetectSpeech(window, state, TestAudio.SampleRate);
            state.Reset();

            Assert.Equal(
                model.DetectSpeech(window, fresh, TestAudio.SampleRate),
                model.DetectSpeech(window, state, TestAudio.SampleRate));
        }

        [Fact]
        public void DisposedModel_Throws()
        {
            var model = new VadModel(TestAudio.V5ModelPath);
            var state = model.CreateState();

            model.Dispose();
            model.Dispose(); // a second call is harmless

            Assert.Throws<ObjectDisposedException>(() => model.CreateState());
            Assert.Throws<ObjectDisposedException>(() => model.DetectSpeech(new float[512], state, TestAudio.SampleRate));
        }

        [Fact]
        public void CreateState_RejectsInvalidBatchSize()
        {
            using var model = new VadModel(TestAudio.V5ModelPath);

            Assert.Throws<ArgumentOutOfRangeException>(() => model.CreateState(TestAudio.SampleRate, 0));
        }

        [Fact]
        public void StateReset_ClearsRecurrentState()
        {
            using var model = new VadModel(TestAudio.V5ModelPath);
            var window = TestAudio.Speech.AsSpan(0, 512);

            var state = model.CreateState(TestAudio.SampleRate);
            var fresh = model.CreateState(TestAudio.SampleRate);

            model.DetectSpeech(window, state, TestAudio.SampleRate);
            state.Reset();
            var reset = model.DetectSpeech(window, state, TestAudio.SampleRate);
            var fromFresh = model.DetectSpeech(window, fresh, TestAudio.SampleRate);

            Assert.Equal(fromFresh, reset);
        }

        [Fact]
        public void DetectsSilence_AfterProcessingSpeech()
        {
            using var model = new VadModel(TestAudio.V5ModelPath);
            var state = model.CreateState(TestAudio.SampleRate);

            for (var offset = 0; offset + 512 <= TestAudio.Speech.Length; offset += 512)
            {
                model.DetectSpeech(TestAudio.Speech.AsSpan(offset, 512), state, TestAudio.SampleRate);
            }

            var silenceProbability = model.DetectSpeech(new float[512], state, TestAudio.SampleRate);

            Assert.True(silenceProbability < 0.5f, $"Expected silence after speech, got {silenceProbability}.");
        }
    }

    public class SileroVadModelTests
    {
        [Fact]
        public void LegacyApi_StillProducesProbabilities()
        {
            using var model = new SileroVadModel(File.ReadAllBytes(TestAudio.V4ModelPath));
            var (h, c, sr) = SileroVadModel.GetInitialStateTensors(batchSize: 1, sampleRate: TestAudio.SampleRate);

            Assert.Equal(new[] { 2, 1, 64 }, ((DenseTensor<float>)h).Dimensions.ToArray());
            Assert.Equal(new[] { 2, 1, 64 }, ((DenseTensor<float>)c).Dimensions.ToArray());

            var state = (h, c);
            var probabilities = new List<float>();

            for (var offset = 0; offset + 1024 <= TestAudio.Speech.Length; offset += 1024)
            {
                var (probability, next) = model.DetectSpeech(
                    TestAudio.Speech.AsSpan(offset, 1024),
                    state,
                    sr,
                    batchSize: 1);

                state = next;
                probabilities.Add(probability);
            }

            Assert.NotEmpty(probabilities);
            Assert.All(probabilities, p => Assert.InRange(p, 0f, 1f));
            Assert.Contains(probabilities, p => p >= 0.5f);
        }

        [Fact]
        public void LegacyApi_RejectsV5Model()
        {
            var bytes = File.ReadAllBytes(TestAudio.V5ModelPath);

            var exception = Assert.Throws<NotSupportedException>(() => new SileroVadModel(bytes));

            Assert.Contains("VadModel", exception.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void LegacyApi_RejectsGarbageModel()
        {
            Assert.ThrowsAny<Exception>(() => new SileroVadModel(new byte[] { 1, 2, 3, 4 }));
        }
    }
}
