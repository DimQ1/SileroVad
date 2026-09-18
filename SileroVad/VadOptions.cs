namespace SileroVad
{
    /// <summary>
    /// Tunable options for speech detection with <see cref="Vad"/> and <see cref="VadStream"/>.
    /// </summary>
    /// <remarks>
    /// The defaults match the historical <see cref="Vad.GetSpeechTimestamps(System.ReadOnlySpan{float}, float, int, float, int, int, int)"/>
    /// behaviour, so switching to the options based overloads does not change results.
    /// </remarks>
    public sealed record VadOptions
    {
        /// <summary>
        /// Sample rate of the audio that is analysed. Supported values are 8000, 16000 and any multiple of
        /// 16000 (audio is decimated to 16000 before inference and reported positions are scaled back to
        /// the original sample rate).
        /// </summary>
        public int SampleRate { get; init; } = 16000;

        /// <summary>
        /// Speech threshold: probabilities above this value count as speech. Values between
        /// <see cref="Threshold"/> <c>- 0.15</c> and <see cref="Threshold"/> keep the current segment open.
        /// </summary>
        public float Threshold { get; init; } = 0.5f;

        /// <summary>Speech chunks shorter than this are discarded.</summary>
        public int MinSpeechDurationMs { get; init; } = 50;

        /// <summary>
        /// Maximum length of a single speech chunk in seconds. Longer chunks are split at the last sufficiently
        /// long silence when possible and are cut aggressively at the limit otherwise.
        /// </summary>
        public float MaxSpeechDurationSeconds { get; init; } = float.PositiveInfinity;

        /// <summary>Silence duration that closes the current speech chunk.</summary>
        public int MinSilenceDurationMs { get; init; } = 2000;

        /// <summary>
        /// Silence shorter than this is not remembered as a split point when
        /// <see cref="MaxSpeechDurationSeconds"/> forces a cut inside a speech chunk.
        /// </summary>
        public int MinSilenceAtMaxSpeechMs { get; init; } = 98;

        /// <summary>
        /// When <see cref="MaxSpeechDurationSeconds"/> is reached, split at the longest silence seen inside the
        /// current chunk instead of at the most recent one.
        /// </summary>
        /// <remarks>
        /// This mirrors the current default of the reference implementation. It is off by default so that results
        /// stay identical to those of previous library versions; it only takes effect for audio longer than
        /// <see cref="MaxSpeechDurationSeconds"/>, which defaults to infinity.
        /// </remarks>
        public bool UseLongestSilenceForMaxSpeech { get; init; }

        /// <summary>
        /// Number of samples fed to the model per inference. <see langword="null"/> selects the recommended
        /// window for the loaded model and sample rate. v5 models only accept their fixed window size
        /// (512 samples at 16 kHz, 256 at 8 kHz), so the value is ignored for them.
        /// </summary>
        public int? WindowSizeSamples { get; init; }

        /// <summary>Speech chunks are padded by this much on both sides.</summary>
        public int SpeechPadMs { get; init; } = 100;

        /// <summary>Options with default values.</summary>
        public static VadOptions Default => new();

        internal void Validate()
        {
            if (Threshold is <= 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(Threshold), Threshold, "Threshold must be in the (0, 1] range.");
            }

            if (MinSpeechDurationMs < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(MinSpeechDurationMs), MinSpeechDurationMs, "Minimum speech duration cannot be negative.");
            }

            if (MinSilenceDurationMs < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(MinSilenceDurationMs), MinSilenceDurationMs, "Minimum silence duration cannot be negative.");
            }

            if (MinSilenceAtMaxSpeechMs < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(MinSilenceAtMaxSpeechMs), MinSilenceAtMaxSpeechMs, "Minimum silence at maximum speech cannot be negative.");
            }

            if (SpeechPadMs < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(SpeechPadMs), SpeechPadMs, "Speech padding cannot be negative.");
            }

            if (WindowSizeSamples is <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(WindowSizeSamples), WindowSizeSamples, "Window size must be positive.");
            }

            if (MaxSpeechDurationSeconds <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(MaxSpeechDurationSeconds), MaxSpeechDurationSeconds, "Maximum speech duration must be positive.");
            }
        }
    }
}
