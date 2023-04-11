﻿using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SileroVad
{
    public class SileroVadModel : IDisposable
    {
        private readonly InferenceSession session;
        private readonly SessionOptions opts;
        private bool disposedValue;


        private SileroVadModel()
        {
            this.opts = new SessionOptions();
            opts.InterOpNumThreads = 1;
            opts.IntraOpNumThreads = 1;
            opts.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
        }

        public SileroVadModel(byte[] model) : this()
        {
            this.session = new InferenceSession(model, this.opts);
        }

        public SileroVadModel(string path) : this()
        {
            this.session = new InferenceSession(path, this.opts);
        }

        public (Tensor<float> h, Tensor<float> c, Tensor<long> sr) GetInitialStateTensors(int batchSize, long sampleRate)
        {
            Tensor<float> h = new DenseTensor<float>(new[] { 2, batchSize, 64 });
            Tensor<float> c = new DenseTensor<float>(new[] { 2, batchSize, 64 });
            Tensor<long> sr = new DenseTensor<long>(new long[] { sampleRate }, new[] { 1 });
            return (h, c, sr);
        }

        public (float, (Tensor<float>, Tensor<float>)) DetectSpeach(ReadOnlySpan<float> x, (Tensor<float>, Tensor<float>) state, Tensor<long> srTensor, int batchSize)
        {
            var (h, c) = state;

            var inputTenzor = ConvertToTensor(x, x.Length);

            List<NamedOnnxValue> ort_inputs = new()
            {
                NamedOnnxValue.CreateFromTensor("input", inputTenzor),
                NamedOnnxValue.CreateFromTensor("sr", srTensor),
                NamedOnnxValue.CreateFromTensor("h", h),
                NamedOnnxValue.CreateFromTensor("c", c),
            };

            var result = session.Run(ort_inputs).ToArray();

            h = result.FirstOrDefault(r => r.Name == "hn")?.AsTensor<float>() ?? new DenseTensor<float>(new[] { 2, batchSize, 64 });
            c = result.FirstOrDefault(r => r.Name == "cn")?.AsTensor<float>() ?? new DenseTensor<float>(new[] { 2, batchSize, 64 });

            var output = result.FirstOrDefault(r => r.Name == "output")?.AsTensor<float>().ToDenseTensor();

            state = (h, c);
            return (output?.FirstOrDefault() ?? 0L, state);

        }

        private static Tensor<float> ConvertToTensor(ReadOnlySpan<float> inputArray, int inputDimension)
        {
            Tensor<float> input = new DenseTensor<float>(inputArray.ToArray(), new[] { 1, inputDimension });

            return input;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    session.Dispose();
                }

                disposedValue = true;
            }
        }

        ~SileroVadModel()
        {
            session.Dispose();
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