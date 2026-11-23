using Xunit;

namespace SileroVad.Tests
{
    /// <summary>
    /// Segmentation from probabilities: <see cref="Vad.GetSpeechTimestampsFromProbabilities(float[], VadOptions, int?)"/>
    /// and the splitting rules it shares with the batch and streaming APIs.
    /// </summary>
    public class SegmentationTests
    {
        private const int Window = 512;

        public static TheoryData<VadModelKind, int> KindsAndRates => new()
        {
            { VadModelKind.V4, 16000 },
            { VadModelKind.V5, 16000 },
            { VadModelKind.V5, 8000 },
            { VadModelKind.V5, 48000 },
        };

        [Theory]
        [MemberData(nameof(KindsAndRates))]
        public void FromProbabilities_MatchesBatch(VadModelKind kind, int sampleRate)
        {
            using var vad = new Vad(kind);
            var audio = AudioAt(sampleRate);
            var options = new VadOptions { SampleRate = sampleRate, MinSilenceDurationMs = 100 };

            var probabilities = vad.GetSpeechProbabilities(audio, options);

            Assert.Equal(
                vad.GetSpeechTimestamps(audio, options),
                vad.GetSpeechTimestampsFromProbabilities(probabilities, options, audio.Length));
        }

        [Fact]
        public void FromProbabilities_ReusesProbabilitiesForOtherOptions()
        {
            using var vad = new Vad();
            var probabilities = vad.GetSpeechProbabilities(TestAudio.Speech);

            foreach (var threshold in new[] { 0.3f, 0.5f, 0.8f })
            {
                var options = new VadOptions { Threshold = threshold, MinSilenceDurationMs = 100 };

                Assert.Equal(
                    vad.GetSpeechTimestamps(TestAudio.Speech, options),
                    vad.GetSpeechTimestampsFromProbabilities(probabilities, options, TestAudio.Speech.Length));
            }
        }

        [Fact]
        public void FromProbabilities_WithoutLength_AssumesWholeWindows()
        {
            using var vad = new Vad();
            var probabilities = Probabilities((10, 1f));

            var speech = Assert.Single(vad.GetSpeechTimestampsFromProbabilities(probabilities, new VadOptions { SpeechPadMs = 0 }));

            Assert.Equal(new VadSpeech(0, 10 * Window), speech);
        }

        [Fact]
        public void FromProbabilities_Empty_ReturnsNothing()
        {
            using var vad = new Vad();

            Assert.Empty(vad.GetSpeechTimestampsFromProbabilities(Array.Empty<float>(), new VadOptions()));
        }

        [Fact]
        public void FromProbabilities_RejectsNegativeLength()
        {
            using var vad = new Vad();

            Assert.Throws<ArgumentOutOfRangeException>(
                () => vad.GetSpeechTimestampsFromProbabilities(new[] { 1f }, new VadOptions(), -1));
        }

        [Fact]
        public void MaxSpeech_SplitInsideSpeech_KeepsTheRestOfTheSpeech()
        {
            // Continuous speech, no silence to split at: every forced cut has to start the next chunk right away
            // (silero-vad v6.3, snakers4/silero-vad#813). Previously the window at the cut was lost.
            using var vad = new Vad();
            var options = new VadOptions { SpeechPadMs = 0, MaxSpeechDurationSeconds = 0.5f };

            var speeches = vad.GetSpeechTimestampsFromProbabilities(Probabilities((40, 1f)), options);

            Assert.True(speeches.Count > 1);
            Assert.Equal(0, speeches[0].Start);
            Assert.Equal(40 * Window, speeches[^1].End);

            for (var i = 1; i < speeches.Count; i++)
            {
                Assert.Equal(speeches[i - 1].End, speeches[i].Start);
            }
        }

        [Fact]
        public void MaxSpeech_SplitInsidePause_WaitsForSpeech()
        {
            // The limit is reached on a silent window: the next chunk starts when speech returns, not at the cut.
            using var vad = new Vad();
            var options = new VadOptions { SpeechPadMs = 0, MaxSpeechDurationSeconds = 0.5f };

            var speeches = vad.GetSpeechTimestampsFromProbabilities(Probabilities((15, 1f), (2, 0f), (20, 1f)), options);

            Assert.Equal(new VadSpeech(0, 15 * Window), speeches[0]);
            Assert.Equal(17 * Window, speeches[1].Start);
        }

        [Fact]
        public void NegThreshold_DefaultsToThresholdMinus015()
        {
            using var vad = new Vad();
            var probabilities = vad.GetSpeechProbabilities(TestAudio.Speech);
            var implicitOptions = new VadOptions { Threshold = 0.6f, MinSilenceDurationMs = 50 };
            var explicitOptions = implicitOptions with { NegThreshold = 0.6f - 0.15f };

            Assert.Equal(
                vad.GetSpeechTimestampsFromProbabilities(probabilities, implicitOptions),
                vad.GetSpeechTimestampsFromProbabilities(probabilities, explicitOptions));
        }

        [Fact]
        public void NegThreshold_ControlsWhereSpeechEnds()
        {
            // 0.4 is above the default exit threshold (0.35) but below an explicit one of 0.45.
            using var vad = new Vad();
            var probabilities = Probabilities((10, 1f), (10, 0.4f), (10, 1f));
            var options = new VadOptions { SpeechPadMs = 0, MinSilenceDurationMs = 100 };

            Assert.Single(vad.GetSpeechTimestampsFromProbabilities(probabilities, options));
            Assert.Equal(2, vad.GetSpeechTimestampsFromProbabilities(probabilities, options with { NegThreshold = 0.45f }).Count);
        }

        [Theory]
        [InlineData(-0.1f)]
        [InlineData(0.6f)]
        [InlineData(1.5f)]
        public void NegThreshold_IsValidated(float negThreshold)
        {
            using var vad = new Vad();
            var options = new VadOptions { NegThreshold = negThreshold };

            Assert.Throws<ArgumentOutOfRangeException>(() => vad.GetSpeechTimestamps(TestAudio.Silence, options));
        }

        private static float[] AudioAt(int sampleRate) => sampleRate switch
        {
            8000 => TestAudio.Decimate(TestAudio.Speech, 2),
            16000 => TestAudio.Speech,
            _ => TestAudio.Upsample(TestAudio.Speech, sampleRate / 16000),
        };

        private static float[] Probabilities(params (int Count, float Value)[] runs) =>
            runs.SelectMany(run => Enumerable.Repeat(run.Value, run.Count)).ToArray();
    }
}
