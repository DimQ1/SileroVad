using Microsoft.ML.OnnxRuntime.Tensors;

namespace SileroVad
{
    /// <summary>
    /// Legacy, v4 specific wrapper around a Silero VAD ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This type predates <see cref="VadModel"/> and only handles models with separate <c>h</c>/<c>c</c> state
    /// tensors (<see cref="VadModelKind.V4"/>). It is kept for backward compatibility with existing callers;
    /// new code should use <see cref="VadModel"/>, which supports both model revisions.
    /// </para>
    /// <para>
    /// The class is not sealed and keeps its original public surface, including
    /// <see cref="GetInitialStateTensors(int, long)"/> and <see cref="DetectSpeech(System.ReadOnlySpan{float}, (Tensor{float}, Tensor{float}), Tensor{long}, int)"/>.
    /// </para>
    /// </remarks>
    public class SileroVadModel : IDisposable
    {
        private const int HiddenSize = 64;

        private readonly VadModel engine;
        private bool disposedValue;

        /// <summary>Creates a model from ONNX model bytes.</summary>
        /// <param name="model">Raw contents of a Silero VAD v4 <c>.onnx</c> file.</param>
        /// <exception cref="NotSupportedException">The model is not a v4 model.</exception>
        public SileroVadModel(byte[] model)
        {
            this.engine = new VadModel(model);

            if (this.engine.Kind != VadModelKind.V4)
            {
                this.engine.Dispose();
                throw new NotSupportedException(
                    $"SileroVadModel only supports v4 models with '{VadModel.LegacyHiddenInputName}'/'{VadModel.LegacyCellInputName}' state inputs. " +
                    $"Use {nameof(VadModel)} for v5 models.");
            }
        }

        /// <summary>Model revision handled by this instance; always <see cref="VadModelKind.V4"/>.</summary>
        public VadModelKind Kind => VadModelKind.V4;

        /// <summary>Creates zeroed <c>h</c>, <c>c</c> and sample rate tensors for a fresh stream.</summary>
        /// <param name="batchSize">Number of streams processed in lock step.</param>
        /// <param name="sampleRate">Sample rate of the audio.</param>
        public static (Tensor<float> h, Tensor<float> c, Tensor<long> sr) GetInitialStateTensors(int batchSize, long sampleRate)
        {
            Tensor<float> h = new DenseTensor<float>(new[] { 2, batchSize, HiddenSize });
            Tensor<float> c = new DenseTensor<float>(new[] { 2, batchSize, HiddenSize });
            Tensor<long> sr = new DenseTensor<long>(new long[] { sampleRate }, new[] { 1 });
            return (h, c, sr);
        }

        /// <summary>Runs one audio window and returns the speech probability together with the next state.</summary>
        /// <param name="x">Audio window without context. 512, 1024 or 1536 samples are expected at 16 kHz.</param>
        /// <param name="state">Recurrent state of the stream, obtained from a previous call or from <see cref="GetInitialStateTensors(int, long)"/>.</param>
        /// <param name="srTensor">Sample rate tensor.</param>
        /// <param name="batchSize">Number of streams processed in lock step.</param>
        public (float, (Tensor<float>, Tensor<float>)) DetectSpeech(
            ReadOnlySpan<float> x,
            (Tensor<float>, Tensor<float>) state,
            Tensor<long> srTensor,
            int batchSize)
        {
            var sampleRate = (int)srTensor.ToArray()[0];
            var engineState = this.engine.CreateState(sampleRate, batchSize);

            state.Item1.ToArray().CopyTo(engineState.Primary, 0);
            if (engineState.Secondary is not null)
            {
                state.Item2.ToArray().CopyTo(engineState.Secondary, 0);
            }

            var probability = this.engine.DetectSpeech(x, engineState, sampleRate);

            var h = new DenseTensor<float>(engineState.Primary.AsMemory(), engineState.StateDimensions);
            var c = new DenseTensor<float>(engineState.Secondary!.AsMemory(), engineState.StateDimensions);

            return (probability, (h, c));
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    this.engine.Dispose();
                }

                disposedValue = true;
            }
        }

        ~SileroVadModel()
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
