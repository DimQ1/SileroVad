namespace SileroVad
{
    /// <summary>
    /// Helpers for working with the speech segments produced by <see cref="Vad"/>.
    /// </summary>
    public static class VadHelper
    {
        /// <summary>
        /// Concatenates the samples covered by <paramref name="vadSpeeches"/>, clamping segments that fall outside
        /// the audio instead of throwing.
        /// </summary>
        /// <param name="audio">Mono audio samples the segments refer to.</param>
        /// <param name="vadSpeeches">Segments to extract, in ascending order.</param>
        public static IEnumerable<float> GetSpeechSamples(float[] audio, List<VadSpeech> vadSpeeches)
        {
            ArgumentNullException.ThrowIfNull(audio);
            ArgumentNullException.ThrowIfNull(vadSpeeches);
            return GetSpeechSamplesIterator(audio, vadSpeeches);
        }

        /// <summary>
        /// Concatenates the samples covered by <paramref name="vadSpeeches"/>, clamping segments that fall outside
        /// the audio instead of throwing.
        /// </summary>
        /// <param name="audio">Mono audio samples the segments refer to.</param>
        /// <param name="vadSpeeches">Segments to extract; they do not have to be sorted.</param>
        public static IEnumerable<float> GetSpeechSamples(IReadOnlyList<float> audio, IEnumerable<VadSpeech> vadSpeeches)
        {
            ArgumentNullException.ThrowIfNull(audio);
            ArgumentNullException.ThrowIfNull(vadSpeeches);
            return GetSpeechSamplesIterator(audio, vadSpeeches);
        }

        /// <summary>
        /// Returns the samples of <paramref name="audio"/> that are not covered by any speech segment.
        /// </summary>
        /// <param name="audio">Mono audio samples the segments refer to.</param>
        /// <param name="vadSpeeches">Speech segments; they do not have to be sorted.</param>
        public static IEnumerable<float> GetSilenceSamples(float[] audio, IEnumerable<VadSpeech> vadSpeeches)
        {
            ArgumentNullException.ThrowIfNull(audio);
            ArgumentNullException.ThrowIfNull(vadSpeeches);
            return GetSilenceSamplesIterator(audio, vadSpeeches);
        }

        /// <summary>Total duration covered by the given speech segments.</summary>
        /// <param name="vadSpeeches">Speech segments.</param>
        /// <param name="sampleRate">Sample rate the positions are expressed in.</param>
        public static TimeSpan GetSpeechDuration(IEnumerable<VadSpeech> vadSpeeches, int sampleRate)
        {
            ArgumentNullException.ThrowIfNull(vadSpeeches);

            if (sampleRate <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Sample rate must be positive.");
            }

            return TimeSpan.FromSeconds(vadSpeeches.Sum(speech => speech.Length) / (double)sampleRate);
        }

        /// <summary>
        /// Merges segments that are separated by at most <paramref name="maxGapSamples"/> samples.
        /// </summary>
        /// <param name="vadSpeeches">Speech segments; they do not have to be sorted.</param>
        /// <param name="maxGapSamples">Largest gap, in samples, that is still merged.</param>
        public static IReadOnlyList<VadSpeech> Merge(IEnumerable<VadSpeech> vadSpeeches, int maxGapSamples)
        {
            ArgumentNullException.ThrowIfNull(vadSpeeches);

            if (maxGapSamples < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxGapSamples), maxGapSamples, "Maximum gap cannot be negative.");
            }

            var ordered = vadSpeeches.OrderBy(speech => speech.Start).ToList();
            var merged = new List<VadSpeech>(ordered.Count);

            foreach (var speech in ordered)
            {
                if (merged.Count > 0 && speech.Start - merged[^1].End <= maxGapSamples)
                {
                    merged[^1].End = Math.Max(merged[^1].End, speech.End);
                }
                else
                {
                    merged.Add(new VadSpeech(speech.Start, speech.End));
                }
            }

            return merged;
        }

        /// <summary>Projects speech segments into time ranges.</summary>
        /// <param name="vadSpeeches">Speech segments.</param>
        /// <param name="sampleRate">Sample rate the positions are expressed in.</param>
        public static IEnumerable<(TimeSpan Start, TimeSpan End)> ToTimeRanges(
            IEnumerable<VadSpeech> vadSpeeches,
            int sampleRate)
        {
            ArgumentNullException.ThrowIfNull(vadSpeeches);
            return ToTimeRangesIterator(vadSpeeches, sampleRate);
        }

        private static IEnumerable<float> GetSpeechSamplesIterator(
            IReadOnlyList<float> audio,
            IEnumerable<VadSpeech> vadSpeeches)
        {
            foreach (var speech in vadSpeeches)
            {
                var start = Math.Clamp(speech.Start, 0, audio.Count);
                var end = Math.Clamp(speech.End, start, audio.Count);

                for (var i = start; i < end; i++)
                {
                    yield return audio[i];
                }
            }
        }

        private static IEnumerable<float> GetSilenceSamplesIterator(float[] audio, IEnumerable<VadSpeech> vadSpeeches)
        {
            var position = 0;

            foreach (var speech in vadSpeeches.OrderBy(s => s.Start))
            {
                var start = Math.Clamp(speech.Start, position, audio.Length);
                for (var i = position; i < start; i++)
                {
                    yield return audio[i];
                }

                position = Math.Max(position, Math.Clamp(speech.End, 0, audio.Length));
            }

            for (var i = position; i < audio.Length; i++)
            {
                yield return audio[i];
            }
        }

        private static IEnumerable<(TimeSpan Start, TimeSpan End)> ToTimeRangesIterator(
            IEnumerable<VadSpeech> vadSpeeches,
            int sampleRate)
        {
            if (sampleRate <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Sample rate must be positive.");
            }

            foreach (var speech in vadSpeeches)
            {
                yield return (
                    TimeSpan.FromSeconds(speech.Start / (double)sampleRate),
                    TimeSpan.FromSeconds(speech.End / (double)sampleRate));
            }
        }
    }
}
