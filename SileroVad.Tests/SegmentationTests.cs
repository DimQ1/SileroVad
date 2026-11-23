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
        public void MaxSpeech_SplitsAtTheLongestSilence()
        {
            // Pauses of 4, 8 and 6 windows inside speech; the 2 s limit is reached at window 62. Each pause is longer
            // than MinSilenceAtMaxSpeechMs (98 ms) but much shorter than MinSilenceDurationMs, so none closes a chunk.
            using var vad = new Vad();
            var options = MaxSpeechOptions with { UseLongestSilenceForMaxSpeech = true };

            var speeches = vad.GetSpeechTimestampsFromProbabilities(PausedSpeech, options);

            Assert.Equal(
                new[]
                {
                    new VadSpeech(0, 24 * Window),           // cut at the start of the longest (8 window) pause
                    new VadSpeech(32 * Window, 94 * Window), // resumes after it; no pause left at the next limit
                    new VadSpeech(94 * Window, 100 * Window),
                },
                speeches);
        }

        [Fact]
        public void MaxSpeech_SplitsAtTheLastSilenceByDefault()
        {
            // Same audio as above: the legacy rule cuts at the last pause that lasted longer than 98 ms (the 6 window
            // one), not at the longest. It measures a pause while it lasts, so the 4 window pause (96 ms from its
            // first to its last window) does not qualify.
            using var vad = new Vad();

            var speeches = vad.GetSpeechTimestampsFromProbabilities(PausedSpeech, MaxSpeechOptions);

            Assert.Equal(
                new[]
                {
                    new VadSpeech(0, 42 * Window),
                    new VadSpeech(48 * Window, 100 * Window),
                },
                speeches);
        }

        [Theory]
        [InlineData(false, 56)]
        [InlineData(true, 62)]
        public void MaxSpeech_ReachedDuringAPause_WaitsForSpeech(bool useLongestSilence, int expectedCutWindow)
        {
            // The limit (window 62) falls inside a 15 window pause. The legacy rule cuts where the pause began; the
            // longest-silence rule has no finished pause to choose and cuts at the limit. Both resume with speech.
            using var vad = new Vad();
            var options = MaxSpeechOptions with { UseLongestSilenceForMaxSpeech = useLongestSilence };

            var speeches = vad.GetSpeechTimestampsFromProbabilities(Probabilities((56, 1f), (15, 0f), (29, 1f)), options);

            Assert.Equal(
                new[]
                {
                    new VadSpeech(0, expectedCutWindow * Window),
                    new VadSpeech(71 * Window, 100 * Window),
                },
                speeches);
        }

        [Theory]
        [InlineData(new[] { 1, 70 }, new[] { 1f, 0f })]
        [InlineData(new[] { 70, 1 }, new[] { 0f, 1f })]
        public void SpeechShorterThanMinimum_IsDropped(int[] counts, float[] values)
        {
            // One 32 ms window of speech, followed or preceded by silence, is shorter than MinSpeechDurationMs (50 ms).
            using var vad = new Vad();
            var probabilities = Probabilities(counts.Zip(values, (count, value) => (count, value)).ToArray());

            Assert.Empty(vad.GetSpeechTimestampsFromProbabilities(probabilities, new VadOptions { MinSilenceDurationMs = 100 }));
        }

        [Fact]
        public void Padding_SharesAShortGapBetweenNeighbours()
        {
            // The 10 window gap (5120 samples) is shorter than twice the 200 ms pad, so each side takes half of it.
            using var vad = new Vad();
            var options = new VadOptions { MinSilenceDurationMs = 100, SpeechPadMs = 200 };

            var speeches = vad.GetSpeechTimestampsFromProbabilities(Probabilities((10, 1f), (10, 0f), (10, 1f)), options);

            Assert.Equal(new[] { new VadSpeech(0, 15 * Window), new VadSpeech(15 * Window, 30 * Window) }, speeches);
        }

        [Fact]
        public void Padding_IsClampedToTheAudio()
        {
            // Speech from window 2 to 22 ([1024, 11264)), padded by 200 ms (3200 samples): the start would fall
            // before the audio and the end after its 13000 samples, so both are clamped.
            using var vad = new Vad();
            var options = new VadOptions { MinSilenceDurationMs = 100, SpeechPadMs = 200 };
            var probabilities = Probabilities((2, 0f), (20, 1f), (10, 0f));

            var speech = Assert.Single(vad.GetSpeechTimestampsFromProbabilities(probabilities, options, audioLengthSamples: 13000));

            Assert.Equal(new VadSpeech(0, 13000), speech);
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

        /// <summary>No padding and a 2 s limit: 32000 - 512 samples, reached at window 62.</summary>
        private static VadOptions MaxSpeechOptions => new() { SpeechPadMs = 0, MaxSpeechDurationSeconds = 2f };

        /// <summary>100 windows of speech with pauses of 4 (windows 10-13), 8 (24-31) and 6 (42-47) windows.</summary>
        private static float[] PausedSpeech =>
            Probabilities((10, 1f), (4, 0f), (10, 1f), (8, 0f), (10, 1f), (6, 0f), (52, 1f));

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
