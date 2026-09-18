using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SileroVad
{
    /// <summary>
    /// Carries the recurrent state of a single Silero VAD stream (and, for
    /// <see cref="VadModelKind.V5"/> models, the rolling context of the previous window).
    /// </summary>
    /// <remarks>
    /// State objects are opaque: create them through <see cref="VadModel.CreateState(int, int)"/>, or let
    /// <see cref="Vad"/> and <see cref="VadStream"/> manage them for you. Reuse one state per independent
    /// audio stream; call <see cref="Reset"/> to start a new stream.
    /// </remarks>
    public sealed class VadState
    {
        internal VadState(VadModelKind kind, int sampleRate, int batchSize, int stateSize, int contextSamples)
        {
            Kind = kind;
            SampleRate = sampleRate;
            BatchSize = batchSize;
            StateSize = stateSize;
            ContextSamples = contextSamples;

            StateDimensions = new[] { 2, batchSize, stateSize };

            Primary = new float[2 * batchSize * stateSize];
            Secondary = kind == VadModelKind.V4 ? new float[2 * batchSize * stateSize] : null;
            Context = contextSamples > 0 ? new float[batchSize * contextSamples] : Array.Empty<float>();

            PrimaryTensor = new DenseTensor<float>(Primary.AsMemory(), StateDimensions);
            SecondaryTensor = Secondary is null ? null : new DenseTensor<float>(Secondary.AsMemory(), StateDimensions);
            SampleRateTensor = new DenseTensor<long>(new[] { (long)sampleRate }, new[] { 1 });
        }

        /// <summary>Model revision this state belongs to.</summary>
        public VadModelKind Kind { get; }

        /// <summary>Sample rate the state was created for.</summary>
        public int SampleRate { get; }

        /// <summary>Number of independent streams carried by the state.</summary>
        public int BatchSize { get; }

        /// <summary>Width of a single recurrent state tensor (64 for v4, 128 for v5).</summary>
        public int StateSize { get; }

        /// <summary>Samples of context prepended to every window (64 at 16 kHz and 32 at 8 kHz for v5, 0 for v4).</summary>
        public int ContextSamples { get; }

        internal int[] StateDimensions { get; }

        /// <summary>v4 only: the <c>h</c> tensor, or the combined <c>state</c> tensor for v5.</summary>
        internal float[] Primary { get; }

        /// <summary>v4 only: the <c>c</c> tensor; always <see langword="null"/> for v5.</summary>
        internal float[]? Secondary { get; }

        /// <summary>Tail of the previous window, prepended to the next one for v5 models.</summary>
        internal float[] Context { get; }

        internal DenseTensor<float> PrimaryTensor { get; }

        internal DenseTensor<float>? SecondaryTensor { get; }

        internal DenseTensor<long> SampleRateTensor { get; }

        internal float[] InputBuffer { get; private set; } = Array.Empty<float>();

        internal DenseTensor<float>? InputTensor { get; private set; }

        internal List<NamedOnnxValue>? Inputs { get; set; }

        /// <summary>
        /// Returns a writable buffer of <paramref name="length"/> samples to build the next model input in.
        /// </summary>
        internal Span<float> RentInputBuffer(int length)
        {
            if (InputBuffer.Length < length)
            {
                InputBuffer = new float[length];
                InputTensor = null;
                Inputs = null;
            }

            if (InputTensor is null || InputTensor.Dimensions[1] != length)
            {
                InputTensor = new DenseTensor<float>(InputBuffer.AsMemory(0, length), new[] { 1, length });
                Inputs = null;
            }

            return InputBuffer.AsSpan(0, length);
        }

        internal DenseTensor<float> GetInputTensor() =>
            InputTensor ?? throw new InvalidOperationException("No input buffer has been rented yet.");

        /// <summary>Clears the recurrent state and the context, starting a new independent stream.</summary>
        public void Reset()
        {
            Array.Clear(Primary);
            if (Secondary is not null)
            {
                Array.Clear(Secondary);
            }

            if (Context.Length > 0)
            {
                Array.Clear(Context);
            }
        }
    }
}
