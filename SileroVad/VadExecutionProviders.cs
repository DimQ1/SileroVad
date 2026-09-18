using Microsoft.ML.OnnxRuntime;

namespace SileroVad
{
    /// <summary>
    /// Helps selecting the ONNX Runtime execution provider without creating a hard dependency on a specific
    /// provider package.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Reference exactly one runtime package in your application (<c>Microsoft.ML.OnnxRuntime</c> for CPU,
    /// <c>.Gpu</c> for CUDA and TensorRT, <c>.DirectML</c>, <c>.QNN</c>, <c>.EP.WebGpu</c>, ...) and then ask for
    /// a provider here. Requests for providers that the loaded runtime does not offer are ignored unless
    /// <c>throwIfUnavailable</c> is set, so the same configuration works on machines with and without a GPU.
    /// </para>
    /// <para>
    /// Prefer these helpers over a raw <c>SessionOptions.AppendExecutionProvider(name, options)</c> call: the
    /// name based API of ONNX Runtime only accepts a subset of the providers (DirectML, QNN, OpenVINO, WebGPU,
    /// CoreML, ...) and rejects names such as <c>CUDAExecutionProvider</c>; CUDA, TensorRT, ROCm, MIGraphX and
    /// DirectML have to be appended through their dedicated methods, which is what <see cref="Use"/> does.
    /// </para>
    /// <para>
    /// Use the <see cref="VadModel"/> or <see cref="Vad"/> constructor overloads that accept an
    /// <see cref="Action{SessionOptions}"/> to apply the result, or to configure anything else
    /// (thread counts, graph optimisation, ...) that <see cref="SessionOptions"/> offers.
    /// </para>
    /// </remarks>
    public static class VadExecutionProviders
    {
        /// <summary>CPU provider, always available.</summary>
        public const string Cpu = "CPUExecutionProvider";

        /// <summary>CUDA provider (requires <c>Microsoft.ML.OnnxRuntime.Gpu</c> and the CUDA runtime).</summary>
        public const string Cuda = "CUDAExecutionProvider";

        /// <summary>TensorRT provider (requires <c>Microsoft.ML.OnnxRuntime.Gpu</c> and TensorRT).</summary>
        public const string TensorRt = "TensorrtExecutionProvider";

        /// <summary>ROCm provider (requires a runtime package built with ROCm).</summary>
        public const string Rocm = "ROCMExecutionProvider";

        /// <summary>DirectML provider (requires <c>Microsoft.ML.OnnxRuntime.DirectML</c>).</summary>
        public const string DirectMl = "DmlExecutionProvider";

        /// <summary>MIGraphX provider (AMD, requires a runtime package built with MIGraphX).</summary>
        public const string MiGraphX = "MIGraphXExecutionProvider";

        /// <summary>OpenVINO provider (requires a runtime package built with OpenVINO).</summary>
        public const string OpenVino = "OpenVINOExecutionProvider";

        /// <summary>CoreML provider (macOS).</summary>
        public const string CoreMl = "CoreMLExecutionProvider";

        /// <summary>Qualcomm QNN provider (requires <c>Microsoft.ML.OnnxRuntime.QNN</c>).</summary>
        public const string Qnn = "QNNExecutionProvider";

        /// <summary>
        /// WebGPU provider, shipped as a plugin (for example <c>Microsoft.ML.OnnxRuntime.EP.WebGpu</c>) that has
        /// to be registered with <see cref="OrtEnv.RegisterExecutionProviderLibrary"/> before it can be appended.
        /// </summary>
        public const string WebGpu = "WebGpuExecutionProvider";

        /// <summary>Option key holding the device index, for example for CUDA or DirectML.</summary>
        public const string DeviceIdOption = "device_id";

        /// <summary>Option key holding the device type, for example for OpenVINO (<c>AUTO</c>, <c>GPU</c>, ...).</summary>
        public const string DeviceTypeOption = "device_type";

        /// <summary>Checks whether the loaded native runtime offers <paramref name="providerName"/>.</summary>
        /// <param name="providerName">Provider name, for example <see cref="Cuda"/> or <c>cuda</c>.</param>
        public static bool IsAvailable(string providerName)
        {
            if (string.IsNullOrWhiteSpace(providerName))
            {
                return false;
            }

            var canonical = Canonicalize(providerName);
            return VadRuntime.GetAvailableExecutionProviders()
                .Any(available => string.Equals(available, canonical, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Builds a session configuration that appends <paramref name="providerName"/>.
        /// </summary>
        /// <param name="providerName">
        /// Provider to append, for example <see cref="Cuda"/>, <see cref="DirectMl"/> or <c>cuda</c>.
        /// </param>
        /// <param name="providerOptions">
        /// Provider specific options, for example <c>device_id</c> or the OpenVINO <c>device_type</c>.
        /// </param>
        /// <param name="throwIfUnavailable">
        /// When <see langword="true"/>, a provider that the runtime does not offer throws; otherwise the request
        /// is ignored and ONNX Runtime uses the next provider in the session (CPU last), which keeps the same
        /// application working on machines without the required hardware.
        /// </param>
        /// <returns>Configuration to pass to the <see cref="VadModel"/> or <see cref="Vad"/> constructors.</returns>
        public static Action<SessionOptions> Use(
            string providerName,
            IReadOnlyDictionary<string, string>? providerOptions = null,
            bool throwIfUnavailable = false)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(providerName);
            var canonical = Canonicalize(providerName);

            return options =>
            {
                ArgumentNullException.ThrowIfNull(options);

                if (!IsAvailable(canonical))
                {
                    if (throwIfUnavailable)
                    {
                        throw new InvalidOperationException(
                            $"The execution provider '{canonical}' is not offered by the loaded ONNX Runtime. " +
                            $"Available providers: {string.Join(", ", VadRuntime.GetAvailableExecutionProviders())}. " +
                            $"Reference the matching runtime package, for example Microsoft.ML.OnnxRuntime.Gpu for CUDA.");
                    }

                    return;
                }

                Append(options, canonical, providerOptions);
            };
        }

        /// <summary>
        /// Builds a session configuration that adds the CPU provider explicitly, optionally with the memory arena
        /// enabled.
        /// </summary>
        /// <param name="useArena">Whether to use the CPU memory arena (enabled by default in ONNX Runtime).</param>
        public static Action<SessionOptions> UseCpu(bool useArena = true) => options =>
        {
            ArgumentNullException.ThrowIfNull(options);
            options.AppendExecutionProvider_CPU(useArena ? 1 : 0);
        };

        /// <summary>
        /// Appends the provider. CUDA, TensorRT, ROCm, MIGraphX, DirectML, OpenVINO and CPU use their dedicated
        /// ONNX Runtime methods because the generic name based API does not accept them.
        /// </summary>
        private static void Append(SessionOptions options, string provider, IReadOnlyDictionary<string, string>? providerOptions)
        {
            var deviceId = GetOption(providerOptions, DeviceIdOption, defaultValue: 0);

            switch (provider)
            {
                case Cuda:
                    options.AppendExecutionProvider_CUDA(deviceId);
                    break;

                case TensorRt:
                    options.AppendExecutionProvider_Tensorrt(deviceId);
                    break;

                case Rocm:
                    options.AppendExecutionProvider_ROCm(deviceId);
                    break;

                case DirectMl:
                    options.AppendExecutionProvider_DML(deviceId);
                    break;

                case MiGraphX:
                    options.AppendExecutionProvider_MIGraphX(deviceId);
                    break;

                case OpenVino:
                    options.AppendExecutionProvider_OpenVINO(GetOption(providerOptions, DeviceTypeOption, "AUTO"));
                    break;

                case Cpu:
                    options.AppendExecutionProvider_CPU(GetOption(providerOptions, "use_arena", defaultValue: 1));
                    break;

                default:
                    // QNN, WebGPU, SNPE, XNNPACK, CoreML, ... are accepted by name.
                    options.AppendExecutionProvider(
                        provider,
                        providerOptions is null
                            ? new Dictionary<string, string>()
                            : new Dictionary<string, string>(providerOptions, StringComparer.Ordinal));
                    break;
            }
        }

        private static int GetOption(IReadOnlyDictionary<string, string>? providerOptions, string key, int defaultValue) =>
            providerOptions is not null
            && providerOptions.TryGetValue(key, out var value)
            && int.TryParse(value, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var parsed)
                ? parsed
                : defaultValue;

        private static string GetOption(IReadOnlyDictionary<string, string>? providerOptions, string key, string defaultValue) =>
            providerOptions is not null && providerOptions.TryGetValue(key, out var value) && !string.IsNullOrWhiteSpace(value)
                ? value
                : defaultValue;

        /// <summary>Maps short names (the way ONNX Runtime accepts them) to the canonical provider name.</summary>
        private static string Canonicalize(string providerName) => providerName.Trim().ToLowerInvariant() switch
        {
            "cpu" => Cpu,
            "cuda" => Cuda,
            "tensorrt" or "trt" => TensorRt,
            "rocm" => Rocm,
            "dml" or "directml" => DirectMl,
            "migraphx" => MiGraphX,
            "openvino" => OpenVino,
            "coreml" => CoreMl,
            "qnn" => Qnn,
            "webgpu" => WebGpu,
            _ => providerName.Trim(),
        };
    }
}
