namespace SileroVad
{
    /// <summary>
    /// Incremental voice activity detector: push audio in arbitrary chunks as it arrives and receive the speech
    /// segments that are already complete.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The stream keeps its own recurrent state, so it can be fed continuously (microphone, network call, ...).
    /// It does not own the <see cref="Vad"/> it was created from: dispose the stream before the detector.
    /// </para>
    /// <para>
    /// Compared to <see cref="Vad.GetSpeechTimestamps(System.ReadOnlySpan{float}, VadOptions)"/> the streaming API
    /// pads every segment on its own instead of sharing the silence between segments that are closer than twice
    /// <see cref="VadOptions.SpeechPadMs"/>; segments are also clipped to the audio pushed so far. Both differences
    /// only matter for speech chunks that follow each other within <c>2 * SpeechPadMs</c> or that hit
    /// <see cref="VadOptions.MaxSpeechDurationSeconds"/>.
    /// </para>
    /// </remarks>
    public sealed class VadStream : IDisposable
    {
        private readonly Vad vad;
        private readonly VadOptions options;
        private readonly VadState state;
        private readonly int windowSize;
        private readonly int step;
        private readonly float[] pending;
        private readonly List<VadSpeech> closed = new();

        private VadSegmenter segmenter;
        private int pendingCount;
        private int windowIndex;
        private int modelSamples;
        private int decimationPhase;
        private bool disposed;

        internal VadStream(Vad vad, VadOptions options)
        {
            this.vad = vad ?? throw new ArgumentNullException(nameof(vad));
            options.Validate();
            this.options = options;

            var modelRate = VadModel.EffectiveSampleRate(options.SampleRate);
            this.step = options.SampleRate / modelRate;
            this.windowSize = vad.Model.IsWindowSizeFixed
                ? vad.Model.GetWindowSamples(modelRate)
                : options.WindowSizeSamples ?? vad.Model.GetWindowSamples(modelRate);

            this.state = vad.Model.CreateState(modelRate);
            this.segmenter = new VadSegmenter(this.windowSize, modelRate, options);
            this.pending = new float[this.windowSize];
        }

        /// <summary>Sample rate of the audio pushed into the stream.</summary>
        public int SampleRate => options.SampleRate;

        /// <summary>Number of samples fed to the model per inference.</summary>
        public int WindowSizeSamples => windowSize;

        /// <summary>
        /// Factor the incoming audio is decimated by before inference: 1 for native rates, otherwise
        /// <see cref="SampleRate"/> / 16000.
        /// </summary>
        public int DecimationFactor => step;

        /// <summary>Number of input samples pushed into the stream so far.</summary>
        public long SamplesProcessed => (long)modelSamples * step;

        /// <summary>
        /// Adds audio to the stream and returns the speech segments completed by it.
        /// </summary>
        /// <param name="samples">
        /// Mono audio samples at <see cref="SampleRate"/>. Any length is accepted; positions reported by this
        /// stream are measured in samples of the concatenation of everything pushed so far.
        /// </param>
        public IReadOnlyList<VadSpeech> Push(ReadOnlySpan<float> samples)
        {
            ThrowIfDisposed();

            if (samples.Length == 0)
            {
                return Array.Empty<VadSpeech>();
            }

            var completed = new List<VadSpeech>();

            foreach (var sample in samples)
            {
                if (decimationPhase == 0)
                {
                    AppendSample(sample, completed);
                }

                decimationPhase++;
                if (decimationPhase == step)
                {
                    decimationPhase = 0;
                }
            }

            return completed;
        }

        /// <summary>
        /// Ends the stream: the trailing partial window is processed (zero padded, like the batch API) and the
        /// still open segment, if any, is closed. Call <see cref="Reset"/> to start a new stream afterwards.
        /// </summary>
        /// <returns>The speech segments completed by the end of the stream.</returns>
        public IReadOnlyList<VadSpeech> Flush()
        {
            ThrowIfDisposed();

            var completed = new List<VadSpeech>();

            if (pendingCount > 0)
            {
                Array.Clear(pending, pendingCount, windowSize - pendingCount);
                pendingCount = 0;
                ProcessWindow(completed);
            }

            closed.Clear();
            segmenter.Complete(modelSamples, closed);
            foreach (var speech in closed)
            {
                completed.Add(ToReported(speech));
            }

            return completed;
        }

        /// <summary>Clears the recurrent state and the buffered audio, starting a new independent stream.</summary>
        public void Reset()
        {
            ThrowIfDisposed();

            state.Reset();
            pendingCount = 0;
            windowIndex = 0;
            modelSamples = 0;
            decimationPhase = 0;
            segmenter = new VadSegmenter(windowSize, VadModel.EffectiveSampleRate(options.SampleRate), options);
        }

        private void AppendSample(float sample, List<VadSpeech> completed)
        {
            pending[pendingCount++] = sample;
            modelSamples++;

            if (pendingCount == windowSize)
            {
                ProcessWindow(completed);
                pendingCount = 0;
            }
        }

        private void ProcessWindow(List<VadSpeech> completed)
        {
            var probability = vad.DetectWindow(pending, state);

            closed.Clear();
            segmenter.Append(probability, windowIndex, closed);
            windowIndex++;

            foreach (var speech in closed)
            {
                completed.Add(ToReported(speech));
            }
        }

        private VadSpeech ToReported(VadSpeech speech)
        {
            var pad = segmenter.SpeechPadSamples;
            var start = Math.Max(0, speech.Start - pad);
            var end = Math.Min(modelSamples, speech.End + pad);
            return new VadSpeech(start * step, end * step);
        }

        private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);

        /// <inheritdoc />
        public void Dispose() => disposed = true;
    }
}
