using Microsoft.ML.OnnxRuntime;
using SileroVad.Properties;

namespace SileroVad
{
    /// <summary>
    /// Voice activity detector built on top of a Silero VAD ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// By default the instance uses the latest bundled model (v5). Pass <see cref="VadModelKind.V4"/> or your
    /// own model bytes / file to use a different model.
    /// </para>
    /// <para>
    /// The detector is stateless between calls: every call to <see cref="GetSpeechTimestamps(System.ReadOnlySpan{float}, VadOptions)"/>
    /// starts from a fresh recurrent state. Use <see cref="CreateStream"/> for incremental processing.
    /// </para>
    /// <para>
    /// Type and member names follow the historical API; the snake_case parameter names of
    /// <see cref="GetSpeechTimestamps(System.ReadOnlySpan{float}, float, int, float, int, int, int)"/> are kept for
    /// backward compatibility and cannot be renamed without breaking named arguments.
    /// </para>
    /// </remarks>
    public class Vad : IDisposable
    {
        private readonly VadModel _model;
        private readonly bool _ownsModel;
        private bool disposedValue;

        /// <summary>Creates a detector using the bundled v5 model.</summary>
        public Vad() : this(VadModelKind.V5)
        {
        }

        /// <summary>Creates a detector using one of the bundled models.</summary>
        /// <param name="kind">Bundled model revision: <see cref="VadModelKind.V5"/> (default) or <see cref="VadModelKind.V4"/>.</param>
        public Vad(VadModelKind kind) : this(kind, null)
        {
        }

        /// <summary>Creates a detector using one of the bundled models and configures the inference session.</summary>
        /// <param name="kind">Bundled model revision: <see cref="VadModelKind.V5"/> (default) or <see cref="VadModelKind.V4"/>.</param>
        /// <param name="configureSession">
        /// Called with the <see cref="Microsoft.ML.OnnxRuntime.SessionOptions"/> before the session is created;
        /// use it to append an execution provider (see <see cref="VadExecutionProviders"/>) or to tune ONNX
        /// Runtime settings.
        /// </param>
        public Vad(VadModelKind kind, Action<SessionOptions>? configureSession)
        {
            var model = kind switch
            {
                VadModelKind.V5 => Resources.silero_vad,
                VadModelKind.V4 => Resources.silero_vad_v4,
                _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unsupported model revision."),
            };

            this._model = new VadModel(model, configureSession);
            this._ownsModel = true;
        }

        /// <summary>Creates a detector using ONNX model bytes.</summary>
        /// <param name="model">Raw contents of a Silero VAD <c>.onnx</c> file (v4 or v5).</param>
        public Vad(byte[] model) : this(model, null)
        {
        }

        /// <summary>Creates a detector using ONNX model bytes and configures the inference session.</summary>
        /// <param name="model">Raw contents of a Silero VAD <c>.onnx</c> file (v4 or v5).</param>
        /// <param name="configureSession">
        /// Called with the <see cref="Microsoft.ML.OnnxRuntime.SessionOptions"/> before the session is created.
        /// </param>
        public Vad(byte[] model, Action<SessionOptions>? configureSession)
        {
            this._model = new VadModel(model, configureSession);
            this._ownsModel = true;
        }

        /// <summary>Creates a detector using an ONNX model file.</summary>
        /// <param name="modelPath">Path to a Silero VAD <c>.onnx</c> file (v4 or v5).</param>
        public Vad(string modelPath) : this(modelPath, null)
        {
        }

        /// <summary>Creates a detector using an ONNX model file and configures the inference session.</summary>
        /// <param name="modelPath">Path to a Silero VAD <c>.onnx</c> file (v4 or v5).</param>
        /// <param name="configureSession">
        /// Called with the <see cref="Microsoft.ML.OnnxRuntime.SessionOptions"/> before the session is created.
        /// </param>
        public Vad(string modelPath, Action<SessionOptions>? configureSession)
        {
            this._model = new VadModel(modelPath, configureSession);
            this._ownsModel = true;
        }

        /// <summary>Creates a detector using an existing model instance.</summary>
        /// <param name="model">
        /// Model to use. The model is <b>not</b> disposed by this instance, so it can be shared between
        /// detectors as long as calls are serialized.
        /// </param>
        public Vad(VadModel model)
        {
            this._model = model ?? throw new ArgumentNullException(nameof(model));
            this._ownsModel = false;
        }

        /// <summary>Revision of the model in use.</summary>
        public VadModelKind ModelKind => _model.Kind;

        /// <summary>Model backing this detector.</summary>
        internal VadModel Model => _model;

        /// <summary>
        /// Splits audio into speech chunks using silero VAD.
        /// </summary>
        /// <remarks>
        /// This is the historical API and always works at 16 kHz. For v5 models the window is fixed by the model
        /// (512 samples), so <paramref name="window_size_samples"/> is only honoured for v4 models.
        /// </remarks>
        /// <param name="audio">Mono audio samples at 16 kHz.</param>
        /// <param name="threshold">
        /// Speech threshold. Probabilities above this value are considered as speech; it is better to tune this
        /// parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
        /// </param>
        /// <param name="min_speech_duration_ms">Final speech chunks shorter than this are thrown out.</param>
        /// <param name="max_speech_duration_s">
        /// Maximum duration of speech chunks in seconds. Chunks longer than that are split at the timestamp of the
        /// last silence that lasts more than 98 ms (if any), to prevent aggressive cutting. Otherwise they are
        /// split aggressively just before the limit.
        /// </param>
        /// <param name="min_silence_duration_ms">At the end of each speech chunk wait this long before separating it.</param>
        /// <param name="window_size_samples">
        /// Audio chunks of this size are fed to the model.
        /// WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate.
        /// Values other than these may affect model performance!
        /// </param>
        /// <param name="speech_pad_ms">Final speech chunks are padded by this much on each side.</param>
        /// <returns>Speech chunks with start and end positions expressed in samples.</returns>
        public List<VadSpeech> GetSpeechTimestamps(
            ReadOnlySpan<float> audio,
            float threshold = 0.5f,
            int min_speech_duration_ms = 50,
            float max_speech_duration_s = float.PositiveInfinity,
            int min_silence_duration_ms = 2000,
            int window_size_samples = 1024,
            int speech_pad_ms = 100)
        {
            if (!new List<int> { 512, 1024, 1536 }.Contains(window_size_samples))
            {
                Console.WriteLine("Unusual window_size_samples! Supported window_size_samples:\n - [512, 1024, 1536] for 16000 sampling_rate");
            }

            var options = new VadOptions
            {
                SampleRate = 16000,
                Threshold = threshold,
                MinSpeechDurationMs = min_speech_duration_ms,
                MaxSpeechDurationSeconds = max_speech_duration_s,
                MinSilenceDurationMs = min_silence_duration_ms,
                WindowSizeSamples = window_size_samples,
                SpeechPadMs = speech_pad_ms,
            };

            return GetSpeechTimestampsCore(audio, options);
        }

        /// <summary>
        /// Splits audio into speech chunks using silero VAD.
        /// </summary>
        /// <param name="audio">Mono audio samples; the rate is described by <see cref="VadOptions.SampleRate"/>.</param>
        /// <param name="options">Detection options.</param>
        /// <returns>Speech chunks with start and end positions expressed in samples of <paramref name="audio"/>.</returns>
        public List<VadSpeech> GetSpeechTimestamps(ReadOnlySpan<float> audio, VadOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);
            return GetSpeechTimestampsCore(audio, options);
        }

        /// <summary>
        /// Splits audio into speech chunks using silero VAD.
        /// </summary>
        /// <param name="audio">Mono audio samples; the rate is described by <see cref="VadOptions.SampleRate"/>.</param>
        /// <param name="options">Detection options.</param>
        /// <returns>Speech chunks with start and end positions expressed in samples of <paramref name="audio"/>.</returns>
        public List<VadSpeech> GetSpeechTimestamps(float[] audio, VadOptions options)
            => GetSpeechTimestamps((ReadOnlySpan<float>)audio, options);

        /// <summary>
        /// Runs the model over the audio and returns the raw speech probability of every window.
        /// </summary>
        /// <param name="audio">Mono audio samples; the rate is described by <see cref="VadOptions.SampleRate"/>.</param>
        /// <param name="options">
        /// Detection options. Only <see cref="VadOptions.SampleRate"/> and <see cref="VadOptions.WindowSizeSamples"/>
        /// influence the probabilities.
        /// </param>
        /// <returns>One probability per window, in chronological order.</returns>
        public float[] GetSpeechProbabilities(ReadOnlySpan<float> audio, VadOptions? options = null)
        {
            options ??= VadOptions.Default;
            options.Validate();

            var decimated = PrepareAudio(audio, options, out var modelRate, out _);
            var signal = decimated is null ? audio : decimated.AsSpan();

            return ComputeProbabilities(signal, ResolveWindowSize(options, modelRate), modelRate);
        }

        /// <summary>
        /// Runs the model over the audio and returns the raw speech probability of every window.
        /// </summary>
        /// <param name="audio">Mono audio samples; the rate is described by <see cref="VadOptions.SampleRate"/>.</param>
        /// <param name="options">
        /// Detection options. Only <see cref="VadOptions.SampleRate"/> and <see cref="VadOptions.WindowSizeSamples"/>
        /// influence the probabilities.
        /// </param>
        /// <returns>One probability per window, in chronological order.</returns>
        public float[] GetSpeechProbabilities(float[] audio, VadOptions? options = null)
            => GetSpeechProbabilities((ReadOnlySpan<float>)audio, options);

        /// <summary>
        /// Creates a stateful detector that processes audio incrementally, as it arrives.
        /// </summary>
        /// <param name="options">Detection options; <see cref="VadOptions.SampleRate"/> describes the pushed audio.</param>
        /// <returns>
        /// A stream that owns its own recurrent state. The stream must not outlive this detector and must be
        /// disposed before it.
        /// </returns>
        public VadStream CreateStream(VadOptions? options = null) => new(this, options ?? VadOptions.Default);

        /// <summary>Runs a single window through the model, advancing <paramref name="state"/>.</summary>
        internal float DetectWindow(ReadOnlySpan<float> window, VadState state) =>
            _model.DetectSpeech(window, state, state.SampleRate);

        private List<VadSpeech> GetSpeechTimestampsCore(ReadOnlySpan<float> audio, VadOptions options)
        {
            options.Validate();

            var decimated = PrepareAudio(audio, options, out var modelRate, out var step);
            var signal = decimated is null ? audio : decimated.AsSpan();
            var windowSize = ResolveWindowSize(options, modelRate);
            var probabilities = ComputeProbabilities(signal, windowSize, modelRate);

            var segmenter = new VadSegmenter(windowSize, modelRate, options);
            var speeches = new List<VadSpeech>();
            for (var i = 0; i < probabilities.Length; i++)
            {
                segmenter.Append(probabilities[i], i, speeches);
            }

            segmenter.Complete(signal.Length, speeches);
            VadSegmenter.ApplySpeechPadding(speeches, signal.Length, segmenter.SpeechPadSamples);

            if (step > 1)
            {
                // Report positions in the sample rate of the caller's audio.
                foreach (var speech in speeches)
                {
                    speech.Start *= step;
                    speech.End *= step;
                }
            }

            return speeches;
        }

        /// <summary>
        /// Prepares the audio for the model: sound rates above 16 kHz are decimated to 16 kHz, the way the
        /// reference implementation does it.
        /// </summary>
        /// <returns>The decimated audio, or <see langword="null"/> when the audio can be used as is.</returns>
        private static float[]? PrepareAudio(
            ReadOnlySpan<float> audio,
            VadOptions options,
            out int modelRate,
            out int step)
        {
            modelRate = VadModel.EffectiveSampleRate(options.SampleRate);
            step = options.SampleRate / modelRate;

            if (step == 1)
            {
                return null;
            }

            var decimated = new float[(audio.Length + step - 1) / step];
            for (int source = 0, target = 0; source < audio.Length; source += step, target++)
            {
                decimated[target] = audio[source];
            }

            return decimated;
        }

        private float[] ComputeProbabilities(ReadOnlySpan<float> audio, int windowSize, int sampleRate)
        {
            var windowCount = (audio.Length + windowSize - 1) / windowSize;
            var probabilities = new float[windowCount];

            if (windowCount == 0)
            {
                return probabilities;
            }

            var state = _model.CreateState(sampleRate);
            var window = new float[windowSize];

            for (var i = 0; i < windowCount; i++)
            {
                var offset = i * windowSize;
                var count = Math.Min(windowSize, audio.Length - offset);

                if (count > 0)
                {
                    audio.Slice(offset, count).CopyTo(window);
                }

                if (count < windowSize)
                {
                    // The trailing partial window is zero padded, as in the reference implementation.
                    Array.Clear(window, count, windowSize - count);
                }

                probabilities[i] = _model.DetectSpeech(window, state, sampleRate);
            }

            return probabilities;
        }

        /// <summary>
        /// Resolves the window size to use. v5 models only accept one window size per sample rate, so an explicit
        /// value is ignored for them.
        /// </summary>
        private int ResolveWindowSize(VadOptions options, int sampleRate)
        {
            var modelWindow = _model.GetWindowSamples(sampleRate);
            return _model.IsWindowSizeFixed ? modelWindow : options.WindowSizeSamples ?? modelWindow;
        }

        /// <summary>Disposes the model when this instance created it.</summary>
        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing && _ownsModel)
                {
                    this._model.Dispose();
                }

                disposedValue = true;
            }
        }

        // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        ~Vad()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
