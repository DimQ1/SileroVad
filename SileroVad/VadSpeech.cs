namespace SileroVad
{
    /// <summary>
    /// A speech segment located inside the analysed audio.
    /// </summary>
    public record VadSpeech
    {
        /// <summary>Creates an empty segment.</summary>
        public VadSpeech()
        {
        }

        /// <summary>Creates a segment spanning <paramref name="start"/> (inclusive) to <paramref name="end"/> (exclusive).</summary>
        /// <param name="start">First sample of the segment.</param>
        /// <param name="end">One past the last sample of the segment.</param>
        public VadSpeech(int start, int end)
        {
            Start = start;
            End = end;
        }

        /// <summary>First sample of the segment.</summary>
        public int Start { get; set; }

        /// <summary>One past the last sample of the segment.</summary>
        public int End { get; set; }

        /// <summary>Number of samples in the segment.</summary>
        public int Length => End - Start;

        /// <summary>Duration of the segment at the given sample rate.</summary>
        /// <param name="sampleRate">Sample rate the positions are expressed in.</param>
        public TimeSpan Duration(int sampleRate) => TimeSpan.FromSeconds(Length / (double)sampleRate);

        /// <summary>Checks whether <paramref name="sampleIndex"/> falls inside the segment.</summary>
        public bool Contains(int sampleIndex) => sampleIndex >= Start && sampleIndex < End;

        /// <summary>Checks whether the two segments share at least one sample.</summary>
        public bool Overlaps(VadSpeech other)
        {
            ArgumentNullException.ThrowIfNull(other);
            return Start < other.End && other.Start < End;
        }

        /// <inheritdoc />
        public override string ToString() => $"{Start}..{End}";
    }
}