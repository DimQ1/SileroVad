using Xunit;

namespace SileroVad.Tests
{
    public class VadSpeechTests
    {
        [Fact]
        public void Length_IsTheSampleCount()
        {
            var speech = new VadSpeech(100, 250);

            Assert.Equal(150, speech.Length);
        }

        [Fact]
        public void Duration_UsesTheSampleRate()
        {
            var speech = new VadSpeech(0, 8000);

            Assert.Equal(TimeSpan.FromSeconds(0.5), speech.Duration(16000));
        }

        [Theory]
        [InlineData(99, false)]
        [InlineData(100, true)]
        [InlineData(249, true)]
        [InlineData(250, false)]
        public void Contains_ExcludesTheEnd(int sampleIndex, bool expected)
        {
            var speech = new VadSpeech(100, 250);

            Assert.Equal(expected, speech.Contains(sampleIndex));
        }

        [Theory]
        [InlineData(0, 50, false)]
        [InlineData(0, 100, false)]
        [InlineData(99, 101, true)]
        [InlineData(249, 300, true)]
        [InlineData(250, 300, false)]
        public void Overlaps_DetectsSharedSamples(int start, int end, bool expected)
        {
            var speech = new VadSpeech(100, 250);

            Assert.Equal(expected, speech.Overlaps(new VadSpeech(start, end)));
        }

        [Fact]
        public void Records_CompareByValue()
        {
            Assert.Equal(new VadSpeech(1, 2), new VadSpeech(1, 2));
            Assert.NotEqual(new VadSpeech(1, 2), new VadSpeech(1, 3));
            Assert.Equal("1..2", new VadSpeech(1, 2).ToString());
        }
    }

    public class VadHelperTests
    {
        private static readonly float[] Audio = Enumerable.Range(0, 100).Select(i => (float)i).ToArray();

        [Fact]
        public void GetSpeechSamples_ConcatenatesSegments()
        {
            var segments = new List<VadSpeech> { new(0, 3), new(10, 12) };

            Assert.Equal(new[] { 0f, 1f, 2f, 10f, 11f }, VadHelper.GetSpeechSamples(Audio, segments));
        }

        [Fact]
        public void GetSpeechSamples_ClampsSegmentsOutsideTheAudio()
        {
            var segments = new List<VadSpeech> { new(-5, 2), new(98, 200) };

            Assert.Equal(
                new[] { 0f, 1f, 98f, 99f },
                VadHelper.GetSpeechSamples(Audio, segments));
        }

        [Fact]
        public void GetSilenceSamples_ReturnsTheComplement()
        {
            var segments = new List<VadSpeech> { new(2, 5) };

            var silence = VadHelper.GetSilenceSamples(Audio, segments).ToArray();

            Assert.Equal(97, silence.Length);
            Assert.Equal(new[] { 0f, 1f, 5f, 6f }, silence.Take(4));
            Assert.Equal(99f, silence[^1]);
        }

        [Fact]
        public void Merge_JoinsSegmentsWithinTheGap()
        {
            var segments = new List<VadSpeech> { new(0, 10), new(12, 20), new(40, 50) };

            var merged = VadHelper.Merge(segments, maxGapSamples: 5);

            Assert.Equal(new[] { new VadSpeech(0, 20), new VadSpeech(40, 50) }, merged);
        }

        [Fact]
        public void Merge_KeepsSegmentsApart()
        {
            var segments = new List<VadSpeech> { new(0, 10), new(11, 20) };

            var merged = VadHelper.Merge(segments, maxGapSamples: 0);

            Assert.Equal(2, merged.Count);
        }

        [Fact]
        public void Merge_SortsInput()
        {
            var segments = new List<VadSpeech> { new(40, 50), new(0, 10) };

            var merged = VadHelper.Merge(segments, maxGapSamples: 0);

            Assert.Equal(new[] { new VadSpeech(0, 10), new VadSpeech(40, 50) }, merged);
        }

        [Fact]
        public void Merge_DoesNotMutateTheInput()
        {
            var segments = new List<VadSpeech> { new(0, 10), new(12, 20) };

            VadHelper.Merge(segments, maxGapSamples: 5);

            Assert.Equal(new VadSpeech(0, 10), segments[0]);
        }

        [Fact]
        public void GetSpeechDuration_SumsSegments()
        {
            var segments = new List<VadSpeech> { new(0, 8000), new(9000, 12000) };

            Assert.Equal(TimeSpan.FromSeconds(0.6875), VadHelper.GetSpeechDuration(segments, 16000));
        }

        [Fact]
        public void ToTimeRanges_ConvertsPositions()
        {
            var segments = new List<VadSpeech> { new(0, 8000), new(16000, 24000) };

            var ranges = VadHelper.ToTimeRanges(segments, 16000).ToArray();

            Assert.Equal(2, ranges.Length);
            Assert.Equal(TimeSpan.Zero, ranges[0].Start);
            Assert.Equal(TimeSpan.FromSeconds(0.5), ranges[0].End);
            Assert.Equal(TimeSpan.FromSeconds(1), ranges[1].Start);
            Assert.Equal(TimeSpan.FromSeconds(1.5), ranges[1].End);
        }

        [Fact]
        public void NullArguments_Throw()
        {
            var audio = new float[16];
            var segments = new List<VadSpeech> { new(0, 8) };

            Assert.Throws<ArgumentNullException>(() => VadHelper.GetSpeechSamples((float[])null!, segments));
            Assert.Throws<ArgumentNullException>(() => VadHelper.GetSpeechSamples(audio, (List<VadSpeech>)null!));
            Assert.Throws<ArgumentNullException>(() => VadHelper.GetSpeechSamples((IReadOnlyList<float>)null!, segments));
            Assert.Throws<ArgumentNullException>(() => VadHelper.GetSpeechSamples((IReadOnlyList<float>)audio, (IEnumerable<VadSpeech>)null!));
            Assert.Throws<ArgumentNullException>(() => VadHelper.GetSilenceSamples(null!, segments));
            Assert.Throws<ArgumentNullException>(() => VadHelper.GetSilenceSamples(audio, null!));
            Assert.Throws<ArgumentNullException>(() => VadHelper.GetSpeechDuration(null!, 16000));
            Assert.Throws<ArgumentNullException>(() => VadHelper.Merge(null!, 0));
            Assert.Throws<ArgumentNullException>(() => VadHelper.ToTimeRanges(null!, 16000));
        }

        [Fact]
        public void InvalidNumbers_Throw()
        {
            var segments = new List<VadSpeech> { new(0, 8) };

            Assert.Throws<ArgumentOutOfRangeException>(() => VadHelper.GetSpeechDuration(segments, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => VadHelper.Merge(segments, -1));

            // ToTimeRanges is lazy: the sample rate is checked when the result is enumerated.
            var ranges = VadHelper.ToTimeRanges(segments, 0);
            Assert.Throws<ArgumentOutOfRangeException>(() => ranges.ToList());
        }

        [Fact]
        public void GetSpeechSamples_AcceptsReadOnlyListsAndUnsortedSegments()
        {
            IReadOnlyList<float> audio = Enumerable.Range(0, 10).Select(i => (float)i).ToArray();
            var segments = new[] { new VadSpeech(6, 8), new VadSpeech(1, 3) };

            Assert.Equal(new float[] { 6, 7, 1, 2 }, VadHelper.GetSpeechSamples(audio, segments).ToArray());
        }
    }
}
