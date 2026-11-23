using Microsoft.ML.OnnxRuntime;
using Xunit;

namespace SileroVad.Tests
{
    public class VadRuntimeTests
    {
        [Fact]
        public void NativeRuntime_IsDeployedWithTheApplication()
        {
            VadRuntime.EnsureNativeRuntimeAvailable();

            var path = VadRuntime.ResolveNativeLibraryPath();

            Assert.NotNull(path);
            Assert.True(File.Exists(path), $"'{path}' does not exist.");
            Assert.StartsWith(AppContext.BaseDirectory, path, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void NativeRuntime_IsNotTakenFromTheOperatingSystem()
        {
            var path = VadRuntime.ResolveNativeLibraryPath();

            // A system wide copy (for example the 1.17 build in C:\Windows\System32 that ships with Windows ML)
            // must never be used: it belongs to another version and crashes the process.
            Assert.DoesNotContain(Path.Combine("Windows", "System32"), path!, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void MissingNativeRuntime_IsReportedWithInstructions()
        {
            var emptyDirectory = Path.Combine(Path.GetTempPath(), $"silero-empty-{Guid.NewGuid():N}");
            Directory.CreateDirectory(emptyDirectory);

            try
            {
                Assert.Null(VadRuntime.ResolveNativeLibraryPath(emptyDirectory));

                var exception = Assert.Throws<InvalidOperationException>(
                    () => VadRuntime.EnsureNativeRuntimeAvailable(emptyDirectory));

                Assert.Contains(VadRuntime.NativeLibraryFileName, exception.Message, StringComparison.Ordinal);
                Assert.Contains("Microsoft.ML.OnnxRuntime.Gpu", exception.Message, StringComparison.Ordinal);
                Assert.Contains("Microsoft.ML.OnnxRuntime", exception.Message, StringComparison.Ordinal);
            }
            finally
            {
                Directory.Delete(emptyDirectory, recursive: true);
            }
        }

        [Fact]
        public void AvailableProviders_ContainCpu()
        {
            var providers = VadRuntime.GetAvailableExecutionProviders();

            Assert.Contains(VadExecutionProviders.Cpu, providers);
        }

        [Fact]
        public void UnavailableProvider_IsIgnoredByDefault()
        {
            // The CPU package does not offer CUDA; a configuration that asks for it anyway must not fail.
            var configure = VadExecutionProviders.Use(VadExecutionProviders.Cuda);

            using var vad = new Vad(VadModelKind.V5, configure);

            Assert.NotEmpty(vad.GetSpeechTimestamps(TestAudio.Speech, new VadOptions()));
        }

        [Fact]
        public void UnavailableProvider_ThrowsWhenRequired()
        {
            var unavailable = VadRuntime.GetAvailableExecutionProviders().Contains(VadExecutionProviders.Cuda)
                ? "NotARealExecutionProvider"
                : VadExecutionProviders.Cuda;

            var configure = VadExecutionProviders.Use(unavailable, throwIfUnavailable: true);

            var exception = Assert.Throws<InvalidOperationException>(
                () => new Vad(VadModelKind.V5, configure));

            Assert.Contains(unavailable, exception.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void CpuProviderConfiguration_IsApplied()
        {
            using var vad = new Vad(VadModelKind.V5, VadExecutionProviders.UseCpu());

            Assert.NotEmpty(vad.GetSpeechTimestamps(TestAudio.Speech, new VadOptions()));
        }

        [Fact]
        public void SessionConfiguration_IsApplied()
        {
            var configured = false;

            using var vad = new Vad(VadModelKind.V5, options =>
            {
                configured = true;
                options.IntraOpNumThreads = 2;
                options.EnableMemoryPattern = false;
            });

            Assert.True(configured);
            Assert.NotEmpty(vad.GetSpeechTimestamps(TestAudio.Speech, new VadOptions()));
        }

        [Fact]
        public void ModelConstructor_AcceptsSessionConfiguration()
        {
            Action<SessionOptions> configure = options => options.IntraOpNumThreads = 2;

            using var model = new VadModel(TestAudio.V5ModelPath, configure);
            using var vad = new Vad(model);

            Assert.Equal(VadModelKind.V5, vad.ModelKind);
            Assert.InRange(
                model.DetectSpeech(new float[512], model.CreateState(), TestAudio.SampleRate),
                0f,
                1f);
        }

        [Fact]
        public void ProviderNames_MatchOnnxRuntimeNaming()
        {
            Assert.Equal("CPUExecutionProvider", VadExecutionProviders.Cpu);
            Assert.Equal("CUDAExecutionProvider", VadExecutionProviders.Cuda);
            Assert.Equal("DmlExecutionProvider", VadExecutionProviders.DirectMl);
        }

        [Fact]
        public void ShortProviderNames_AreAccepted()
        {
            Assert.True(VadExecutionProviders.IsAvailable("cpu"));
            Assert.True(VadExecutionProviders.IsAvailable("CPU"));
            Assert.False(VadExecutionProviders.IsAvailable("not-a-provider"));
            Assert.False(VadExecutionProviders.IsAvailable(string.Empty));
        }

        [Fact]
        public void CpuProvider_AcceptsProviderOptions()
        {
            using var vad = new Vad(
                VadModelKind.V5,
                VadExecutionProviders.Use(
                    VadExecutionProviders.Cpu,
                    new Dictionary<string, string> { [VadExecutionProviders.DeviceIdOption] = "0" }));

            Assert.NotEmpty(vad.GetSpeechTimestamps(TestAudio.Speech, new VadOptions()));
        }

        [Fact]
        public void CudaProvider_IsAppliedWhenTheRuntimeOffersIt()
        {
            // CPU-only test hosts skip this; with Microsoft.ML.OnnxRuntime.Gpu the dedicated CUDA API is used,
            // which the generic name based API of ONNX Runtime rejects.
            if (!VadExecutionProviders.IsAvailable(VadExecutionProviders.Cuda))
            {
                return;
            }

            using var vad = new Vad(VadModelKind.V5, VadExecutionProviders.Use(VadExecutionProviders.Cuda));

            Assert.NotEmpty(vad.GetSpeechTimestamps(TestAudio.Speech, new VadOptions()));
        }

        [Fact]
        public void NativeLibraryFileName_IsPlatformSpecific()
        {
            var expected = OperatingSystem.IsWindows()
                ? "onnxruntime.dll"
                : OperatingSystem.IsMacOS() ? "libonnxruntime.dylib" : "libonnxruntime.so";

            Assert.Equal(expected, VadRuntime.NativeLibraryFileName);
        }

        [Fact]
        public void MismatchedNativeRuntime_IsReportedWithBothVersions()
        {
            // Any versioned binary under the native library name stands in for a runtime of another version:
            // the xunit assertion library is 2.x, the managed ONNX Runtime 1.x.
            WithTemporaryDirectory(directory =>
            {
                var fake = Path.Combine(directory, VadRuntime.NativeLibraryFileName);
                File.Copy(typeof(Assert).Assembly.Location, fake);

                Assert.Equal(fake, VadRuntime.ResolveNativeLibraryPath(directory));

                var exception = Assert.Throws<InvalidOperationException>(() => VadRuntime.EnsureNativeRuntimeAvailable(directory));

                Assert.Contains(fake, exception.Message, StringComparison.Ordinal);
                Assert.Contains(VadRuntime.ManagedVersion.ToString(), exception.Message, StringComparison.Ordinal);
            });
        }

        [Fact]
        public void UnversionedNativeRuntime_IsAccepted()
        {
            // A binary without version information cannot be checked; ONNX Runtime decides when it loads it.
            WithTemporaryDirectory(directory =>
            {
                File.WriteAllBytes(Path.Combine(directory, VadRuntime.NativeLibraryFileName), new byte[] { 1, 2, 3, 4 });

                VadRuntime.EnsureNativeRuntimeAvailable(directory);
            });
        }

        [Fact]
        public void NativeRuntime_IsFoundInTheRuntimesFolder()
        {
            WithTemporaryDirectory(directory =>
            {
                var native = Path.Combine(directory, "runtimes", System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier, "native");
                Directory.CreateDirectory(native);
                var library = Path.Combine(native, VadRuntime.NativeLibraryFileName);
                File.WriteAllBytes(library, new byte[] { 1, 2, 3, 4 });

                Assert.Equal(library, VadRuntime.ResolveNativeLibraryPath(directory));
            });
        }

        [Theory]
        [InlineData(" CPU ", true)]
        [InlineData(VadExecutionProviders.Cpu, true)]
        [InlineData("   ", false)]
        [InlineData(null, false)]
        public void IsAvailable_TrimsNamesAndRejectsBlankOnes(string? providerName, bool expected)
        {
            Assert.Equal(expected, VadExecutionProviders.IsAvailable(providerName!));
        }

        [Theory]
        [InlineData("cuda", VadExecutionProviders.Cuda)]
        [InlineData("tensorrt", VadExecutionProviders.TensorRt)]
        [InlineData("trt", VadExecutionProviders.TensorRt)]
        [InlineData("rocm", VadExecutionProviders.Rocm)]
        [InlineData("dml", VadExecutionProviders.DirectMl)]
        [InlineData("DirectML", VadExecutionProviders.DirectMl)]
        [InlineData("migraphx", VadExecutionProviders.MiGraphX)]
        [InlineData("openvino", VadExecutionProviders.OpenVino)]
        [InlineData("coreml", VadExecutionProviders.CoreMl)]
        [InlineData("qnn", VadExecutionProviders.Qnn)]
        [InlineData("webgpu", VadExecutionProviders.WebGpu)]
        public void ShortNames_MapToTheCanonicalProvider(string shortName, string canonical)
        {
            // Only checkable for providers the test host lacks: the error names the canonical provider.
            if (VadExecutionProviders.IsAvailable(canonical))
            {
                return;
            }

            using var options = new SessionOptions();
            var configure = VadExecutionProviders.Use(shortName, throwIfUnavailable: true);

            var exception = Assert.Throws<InvalidOperationException>(() => configure(options));

            Assert.Contains($"'{canonical}'", exception.Message, StringComparison.Ordinal);
        }

        [Theory]
        [InlineData("0")]
        [InlineData("1")]
        [InlineData("not a number")]
        public void CpuProvider_ReadsTheArenaOption(string useArena)
        {
            var configure = VadExecutionProviders.Use("cpu", new Dictionary<string, string> { ["use_arena"] = useArena });

            using var vad = new Vad(VadModelKind.V5, configure);

            Assert.NotEmpty(vad.GetSpeechTimestamps(TestAudio.Speech, new VadOptions()));
        }

        [Fact]
        public void ProviderConfigurations_RejectInvalidArguments()
        {
            Assert.Throws<ArgumentException>(() => VadExecutionProviders.Use("  "));
            Assert.Throws<ArgumentNullException>(() => VadExecutionProviders.Use(VadExecutionProviders.Cpu)(null!));
            Assert.Throws<ArgumentNullException>(() => VadExecutionProviders.UseCpu()(null!));
        }

        [Fact]
        public void CpuProvider_WithoutArena_IsApplied()
        {
            using var vad = new Vad(VadModelKind.V5, VadExecutionProviders.UseCpu(useArena: false));

            Assert.NotEmpty(vad.GetSpeechTimestamps(TestAudio.Speech, new VadOptions()));
        }

        private static void WithTemporaryDirectory(Action<string> test)
        {
            var directory = Path.Combine(Path.GetTempPath(), $"silero-runtime-{Guid.NewGuid():N}");
            Directory.CreateDirectory(directory);

            try
            {
                test(directory);
            }
            finally
            {
                Directory.Delete(directory, recursive: true);
            }
        }
    }
}
