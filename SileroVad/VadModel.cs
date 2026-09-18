using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SileroVad
{
    /// <summary>
    /// Model agnostic wrapper around a Silero VAD ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The model revision is detected from the model's input tensor names, so both the legacy
    /// <see cref="VadModelKind.V4"/> (<c>h</c>/<c>c</c>) and the current <see cref="VadModelKind.V5"/>
    /// (<c>state</c>/<c>stateN</c>) models are supported by the same code.
    /// </para>
    /// <para>
    /// Instances are not thread safe: one model instance can serve many streams, but each stream needs its own
    /// <see cref="VadState"/> and calls for a single state must be serialized.
    /// </para>
    /// </remarks>
    public sealed class VadModel : IDisposable
    {
        internal const string InputName = "input";
        internal const string SampleRateInputName = "sr";
        internal const string OutputName = "output";
        internal const string LegacyHiddenInputName = "h";
        internal const string LegacyCellInputName = "c";
        internal const string StateInputName = "state";
        internal const string LegacyHiddenOutputName = "hn";
        internal const string LegacyCellOutputName = "cn";
        internal const string StateOutputName = "stateN";

        private const int V4StateSize = 64;
        private const int V5StateSize = 128;

        private readonly InferenceSession _session;
        private readonly string[] _outputNames;
        private bool _disposed;

        /// <summary>Creates a model from ONNX model bytes.</summary>
        /// <param name="model">Raw contents of a Silero VAD <c>.onnx</c> file.</param>
        public VadModel(byte[] model)
            : this(model, null)
        {
        }

        /// <summary>Creates a model from ONNX model bytes and configures the inference session.</summary>
        /// <param name="model">Raw contents of a Silero VAD <c>.onnx</c> file.</param>
        /// <param name="configureSession">
        /// Called with the <see cref="SessionOptions"/> right before the session is created. Use it to append an
        /// execution provider (see <see cref="VadExecutionProviders"/>) or to tune threads, graph optimisation
        /// level and other ONNX Runtime settings.
        /// </param>
        public VadModel(byte[] model, Action<SessionOptions>? configureSession)
            : this(CreateSession(model ?? throw new ArgumentNullException(nameof(model)), configureSession))
        {
        }

        /// <summary>Creates a model from an ONNX file on disk.</summary>
        /// <param name="modelPath">Path to a Silero VAD <c>.onnx</c> file.</param>
        public VadModel(string modelPath)
            : this(modelPath, null)
        {
        }

        /// <summary>Creates a model from an ONNX file on disk and configures the inference session.</summary>
        /// <param name="modelPath">Path to a Silero VAD <c>.onnx</c> file.</param>
        /// <param name="configureSession">
        /// Called with the <see cref="SessionOptions"/> right before the session is created. Use it to append an
        /// execution provider (see <see cref="VadExecutionProviders"/>) or to tune threads, graph optimisation
        /// level and other ONNX Runtime settings.
        /// </param>
        public VadModel(string modelPath, Action<SessionOptions>? configureSession)
            : this(CreateSession(
                string.IsNullOrWhiteSpace(modelPath) ? throw new ArgumentException("Model path is required.", nameof(modelPath)) : modelPath,
                configureSession))
        {
        }

        private VadModel(InferenceSession session)
        {
            _session = session;

            var inputs = session.InputMetadata;
            var hasStateInput = inputs.ContainsKey(StateInputName);
            var hasLegacyStateInputs = inputs.ContainsKey(LegacyHiddenInputName) && inputs.ContainsKey(LegacyCellInputName);

            if (!inputs.ContainsKey(InputName) || !inputs.ContainsKey(SampleRateInputName))
            {
                throw new NotSupportedException(
                    $"The model does not look like a Silero VAD model: expected '{InputName}' and '{SampleRateInputName}' inputs, found [{string.Join(", ", inputs.Keys)}].");
            }

            if (hasStateInput && hasLegacyStateInputs)
            {
                throw new NotSupportedException("The model declares both v4 and v5 state inputs, which is not a Silero VAD model layout.");
            }

            if (hasStateInput)
            {
                Kind = VadModelKind.V5;
                StateSize = ReadStateWidth(inputs[StateInputName], V5StateSize);
                _outputNames = new[] { OutputName, StateOutputName };
            }
            else if (hasLegacyStateInputs)
            {
                Kind = VadModelKind.V4;
                StateSize = ReadStateWidth(inputs[LegacyHiddenInputName], V4StateSize);
                _outputNames = new[] { OutputName, LegacyHiddenOutputName, LegacyCellOutputName };
            }
            else
            {
                throw new NotSupportedException(
                    $"Unsupported Silero VAD model revision: expected a '{StateInputName}' input (v5) or '{LegacyHiddenInputName}'/'{LegacyCellInputName}' inputs (v4), found [{string.Join(", ", inputs.Keys)}].");
            }

            if (!session.OutputMetadata.ContainsKey(OutputName))
            {
                throw new NotSupportedException($"The model does not expose the expected '{OutputName}' output.");
            }
        }

        /// <summary>Detected model revision.</summary>
        public VadModelKind Kind { get; }

        /// <summary>Width of a single recurrent state tensor: 64 for v4 models, 128 for v5 models.</summary>
        public int StateSize { get; }

        /// <summary>
        /// <see langword="true"/> when the model only accepts one fixed window size
        /// (<see cref="VadModelKind.V5"/>); <see langword="false"/> for v4 models, which accept
        /// 512, 1024 or 1536 samples at 16 kHz.
        /// </summary>
        public bool IsWindowSizeFixed => Kind == VadModelKind.V5;

        /// <summary>
        /// Checks whether <paramref name="sampleRate"/> can be used: 8000, 16000 or any multiple of 16000
        /// (multiples are decimated to 16000 before inference).
        /// </summary>
        public static bool IsSupportedSampleRate(int sampleRate) =>
            sampleRate == 8000 || (sampleRate >= 16000 && sampleRate % 16000 == 0);

        /// <summary>Maps a supported requested sample rate to the rate the model is actually run with.</summary>
        /// <exception cref="ArgumentOutOfRangeException">The sample rate is not supported.</exception>
        public static int EffectiveSampleRate(int sampleRate)
        {
            if (!IsSupportedSampleRate(sampleRate))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(sampleRate),
                    sampleRate,
                    "Unsupported sample rate. Supported values are 8000, 16000 and multiples of 16000.");
            }

            return sampleRate == 8000 ? 8000 : 16000;
        }

        /// <summary>Number of context samples prepended to every window for the given sample rate.</summary>
        public int GetContextSamples(int sampleRate)
        {
            if (Kind == VadModelKind.V4)
            {
                return 0;
            }

            return EffectiveSampleRate(sampleRate) == 8000 ? 32 : 64;
        }

        /// <summary>
        /// Recommended window size (samples per inference) for the given sample rate: 512 at 16 kHz and
        /// 256 at 8 kHz for v5 models; the historical 1024 (16 kHz) and 256 (8 kHz) for v4 models.
        /// </summary>
        public int GetWindowSamples(int sampleRate)
        {
            var effective = EffectiveSampleRate(sampleRate);
            return effective == 8000 ? 256 : Kind == VadModelKind.V5 ? 512 : 1024;
        }

        /// <summary>Creates a fresh state for one independent audio stream.</summary>
        /// <param name="sampleRate">Sample rate of the audio that will be analysed.</param>
        /// <param name="batchSize">Number of streams to process in lock step.</param>
        public VadState CreateState(int sampleRate = 16000, int batchSize = 1)
        {
            ThrowIfDisposed();

            if (batchSize < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be at least 1.");
            }

            var effective = EffectiveSampleRate(sampleRate);
            return new VadState(Kind, effective, batchSize, StateSize, GetContextSamples(effective));
        }

        /// <summary>
        /// Runs one window through the model and advances <paramref name="state"/>.
        /// </summary>
        /// <param name="chunk">
        /// Audio window without context. For v5 models it must contain exactly
        /// <see cref="GetWindowSamples(int)"/> samples (512 at 16 kHz, 256 at 8 kHz).
        /// </param>
        /// <param name="state">State of the stream, created with <see cref="CreateState(int, int)"/>.</param>
        /// <param name="sampleRate">Sample rate of <paramref name="chunk"/>.</param>
        /// <returns>Speech probability of the window: values above the threshold indicate speech.</returns>
        public float DetectSpeech(ReadOnlySpan<float> chunk, VadState state, int sampleRate)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(state);

            var effective = EffectiveSampleRate(sampleRate);
            if (state.Kind != Kind)
            {
                throw new ArgumentException($"The state was created for a {state.Kind} model but the model is {Kind}.", nameof(state));
            }

            if (state.SampleRate != effective)
            {
                throw new ArgumentException(
                    $"The state was created for {state.SampleRate} Hz but {effective} Hz was requested.",
                    nameof(state));
            }

            if (chunk.Length == 0)
            {
                throw new ArgumentException("The audio window must not be empty.", nameof(chunk));
            }

            if (Kind == VadModelKind.V5 && chunk.Length != GetWindowSamples(effective))
            {
                throw new ArgumentException(
                    $"v5 models require exactly {GetWindowSamples(effective)} samples per call at {effective} Hz, but {chunk.Length} were provided.",
                    nameof(chunk));
            }

            var contextSamples = state.ContextSamples;
            var input = state.RentInputBuffer(chunk.Length + contextSamples);

            if (contextSamples > 0)
            {
                state.Context.AsSpan().CopyTo(input);
            }

            chunk.CopyTo(input[contextSamples..]);

            float probability = 0f;
            var probabilityFound = false;

            using (var results = Run(state))
            {
                foreach (var result in results)
                {
                    switch (result.Name)
                    {
                        case OutputName:
                            probability = ReadFirstValue(result);
                            probabilityFound = true;
                            break;
                        case StateOutputName:
                        case LegacyHiddenOutputName:
                            CopyToBuffer(result, state.Primary);
                            break;
                        case LegacyCellOutputName when state.Secondary is not null:
                            CopyToBuffer(result, state.Secondary);
                            break;
                    }
                }
            }

            if (!probabilityFound)
            {
                throw new InvalidOperationException($"The model did not return the '{OutputName}' output.");
            }

            if (contextSamples > 0)
            {
                input[^contextSamples..].CopyTo(state.Context);
            }

            return probability;
        }

        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(VadState state)
        {
            state.Inputs ??= BuildInputs(state);
            return _session.Run(state.Inputs, _outputNames);
        }

        private List<NamedOnnxValue> BuildInputs(VadState state)
        {
            var inputs = new List<NamedOnnxValue>(Kind == VadModelKind.V5 ? 3 : 4)
            {
                NamedOnnxValue.CreateFromTensor(InputName, state.GetInputTensor()),
                NamedOnnxValue.CreateFromTensor(SampleRateInputName, state.SampleRateTensor),
            };

            if (Kind == VadModelKind.V5)
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(StateInputName, state.PrimaryTensor));
            }
            else
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(LegacyHiddenInputName, state.PrimaryTensor));
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    LegacyCellInputName,
                    state.SecondaryTensor ?? throw new InvalidOperationException("v4 states must carry two tensors.")));
            }

            return inputs;
        }

        private static float ReadFirstValue(DisposableNamedOnnxValue value)
        {
            var tensor = value.AsTensor<float>();
            return tensor is DenseTensor<float> dense ? dense.Buffer.Span[0] : tensor.ToArray()[0];
        }

        private static void CopyToBuffer(DisposableNamedOnnxValue value, float[] destination)
        {
            var tensor = value.AsTensor<float>();
            if (tensor is DenseTensor<float> dense)
            {
                dense.Buffer.Span.CopyTo(destination);
            }
            else
            {
                tensor.ToArray().CopyTo(destination, 0);
            }
        }

        private static int ReadStateWidth(NodeMetadata metadata, int fallback)
        {
            var dimensions = metadata.Dimensions;
            return dimensions.Length > 0 && dimensions[^1] > 0 ? dimensions[^1] : fallback;
        }

        private static SessionOptions CreateSessionOptions() => new()
        {
            InterOpNumThreads = 1,
            IntraOpNumThreads = 1,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL,
        };

        /// <summary>
        /// Creates the session after making sure a native runtime is available and letting the caller adjust the
        /// session options.
        /// </summary>
        private static InferenceSession CreateSession(byte[] model, Action<SessionOptions>? configureSession)
        {
            VadRuntime.EnsureNativeRuntimeAvailable();
            using var options = CreateSessionOptions();
            configureSession?.Invoke(options);
            return new InferenceSession(model, options);
        }

        private static InferenceSession CreateSession(string modelPath, Action<SessionOptions>? configureSession)
        {
            VadRuntime.EnsureNativeRuntimeAvailable();
            using var options = CreateSessionOptions();
            configureSession?.Invoke(options);
            return new InferenceSession(modelPath, options);
        }

        private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

        private void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            if (disposing)
            {
                _session.Dispose();
            }

            _disposed = true;
        }

        /// <inheritdoc />
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
