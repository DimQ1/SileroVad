using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace SileroVad
{
    /// <summary>
    /// Diagnostics for the native ONNX Runtime library that the managed
    /// <c>Microsoft.ML.OnnxRuntime</c> package needs at run time.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The library only depends on the managed ONNX Runtime package (<c>Microsoft.ML.OnnxRuntime.Managed</c>),
    /// so that an application can pick exactly one native runtime and therefore one execution provider family
    /// (CPU, CUDA, DirectML, TensorRT, OpenVINO, ...) without two packages fighting over the same
    /// <c>onnxruntime</c> binary.
    /// </para>
    /// <para>
    /// The price of that freedom is that the application has to reference a runtime package. Without one, the
    /// native library is resolved from the operating system: on Windows, for example,
    /// <c>C:\Windows\System32\onnxruntime.dll</c> (1.17, shipped by Windows ML) is picked up. Loading such a
    /// mismatched build does not throw a catchable exception, it crashes the process, which is why
    /// <see cref="EnsureNativeRuntimeAvailable"/> validates the setup before ONNX Runtime is touched.
    /// </para>
    /// </remarks>
    public static class VadRuntime
    {
        /// <summary>Name of the native ONNX Runtime library on the current platform.</summary>
        public static string NativeLibraryFileName { get; } = ResolveNativeLibraryFileName();

        /// <summary>Version of the managed ONNX Runtime assembly in use (for example 1.30.0.0).</summary>
        public static Version ManagedVersion { get; } = typeof(SessionOptions).Assembly.GetName().Version ?? new Version(0, 0);

        /// <summary>
        /// Locates the native ONNX Runtime library that ships with the application, or <see langword="null"/>
        /// when the application does not carry one.
        /// </summary>
        /// <param name="baseDirectory">Directory to probe; defaults to the application directory.</param>
        /// <returns>Full path of the native library, or <see langword="null"/>.</returns>
        public static string? ResolveNativeLibraryPath(string? baseDirectory = null)
        {
            var directory = string.IsNullOrWhiteSpace(baseDirectory) ? AppContext.BaseDirectory : baseDirectory!;

            foreach (var candidate in GetCandidates(directory))
            {
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            return null;
        }

        /// <summary>
        /// Verifies that a native ONNX Runtime of the same version as the managed assembly is available, so that
        /// a missing or mismatched setup fails with a clear message instead of crashing the process.
        /// </summary>
        /// <param name="baseDirectory">
        /// Directory to probe; defaults to the application directory, which is also the only case where an
        /// already loaded native library is accepted. An explicit directory is checked strictly.
        /// </param>
        /// <exception cref="InvalidOperationException">
        /// No native library was found, or the one found belongs to another ONNX Runtime version.
        /// </exception>
        public static void EnsureNativeRuntimeAvailable(string? baseDirectory = null)
        {
            var path = ResolveNativeLibraryPath(baseDirectory);

            if (path is null)
            {
                if (baseDirectory is null && IsNativeRuntimeLoadedInProcess())
                {
                    // Another component already loaded the right native library into this process.
                    return;
                }

                throw new InvalidOperationException(BuildMissingNativeRuntimeMessage(baseDirectory));
            }

            var nativeVersion = TryGetFileVersion(path);
            if (nativeVersion is not null
                && (nativeVersion.Major != ManagedVersion.Major || nativeVersion.Minor != ManagedVersion.Minor))
            {
                throw new InvalidOperationException(
                    $"The native ONNX Runtime at '{path}' is version {nativeVersion} but the managed one is {ManagedVersion}. " +
                    $"Reference one runtime package with the same version as Microsoft.ML.OnnxRuntime.Managed, " +
                    $"for example Microsoft.ML.OnnxRuntime.Gpu {ManagedVersion.Major}.{ManagedVersion.Minor}.0 instead of {nativeVersion.Major}.{nativeVersion.Minor}.0.");
            }
        }

        /// <summary>
        /// Lists the execution providers that the loaded native ONNX Runtime offers, in the order ONNX Runtime
        /// reports them (CPU is always available).
        /// </summary>
        /// <remarks>Requires a working native runtime; call <see cref="EnsureNativeRuntimeAvailable"/> first.</remarks>
        public static IReadOnlyList<string> GetAvailableExecutionProviders() =>
            OrtEnv.Instance().GetAvailableProviders();

        private static IEnumerable<string> GetCandidates(string directory)
        {
            yield return Path.Combine(directory, NativeLibraryFileName);

            var runtimeIdentifier = RuntimeInformation.RuntimeIdentifier;
            if (!string.IsNullOrEmpty(runtimeIdentifier))
            {
                yield return Path.Combine(directory, "runtimes", runtimeIdentifier, "native", NativeLibraryFileName);
            }

            var onnxAssemblyDirectory = Path.GetDirectoryName(typeof(SessionOptions).Assembly.Location);
            if (!string.IsNullOrEmpty(onnxAssemblyDirectory)
                && !string.Equals(onnxAssemblyDirectory, directory, StringComparison.OrdinalIgnoreCase))
            {
                yield return Path.Combine(onnxAssemblyDirectory, NativeLibraryFileName);
            }
        }

        private static bool IsNativeRuntimeLoadedInProcess()
        {
            try
            {
                var nativeName = NativeLibraryFileName;

                foreach (ProcessModule module in Process.GetCurrentProcess().Modules)
                {
                    // Compare the exact file name: the managed 'Microsoft.ML.OnnxRuntime.dll' is also listed as a
                    // module and must not be mistaken for the native library.
                    if (!string.Equals(Path.GetFileName(module.FileName), nativeName, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    var version = module.FileVersionInfo?.FileVersion;
                    return version is null || MatchesManagedVersion(version);
                }
            }
            catch (Exception)
            {
                // Enumerating modules is best effort; a failure just means we fall back to the file probe.
            }

            return false;
        }

        private static bool MatchesManagedVersion(string fileVersion)
        {
            var nativeVersion = TryParseVersion(fileVersion);
            return nativeVersion is null
                || (nativeVersion.Major == ManagedVersion.Major && nativeVersion.Minor == ManagedVersion.Minor);
        }

        private static Version? TryGetFileVersion(string path)
        {
            try
            {
                return TryParseVersion(FileVersionInfo.GetVersionInfo(path).FileVersion);
            }
            catch (Exception)
            {
                // Not a versioned (or not a readable) binary: accept it and let ONNX Runtime decide.
                return null;
            }
        }

        private static Version? TryParseVersion(string? fileVersion)
        {
            if (string.IsNullOrWhiteSpace(fileVersion))
            {
                return null;
            }

            // Native builds report things like "1.30.20260918-1200.1.os-..."; keep the numeric prefix.
            var trimmed = new string(fileVersion.TakeWhile(c => char.IsDigit(c) || c == '.').ToArray()).TrimEnd('.');
            return Version.TryParse(trimmed, out var version) ? version : null;
        }

        private static string BuildMissingNativeRuntimeMessage(string? baseDirectory)
        {
            var directory = string.IsNullOrWhiteSpace(baseDirectory) ? AppContext.BaseDirectory : baseDirectory!;
            var version = $"{ManagedVersion.Major}.{ManagedVersion.Minor}.0";

            return
                $"No native ONNX Runtime library ('{NativeLibraryFileName}') was found for this application, " +
                $"so the model cannot run. SileroVad only references the managed ONNX Runtime so that you can choose " +
                $"a single provider yourself; add exactly one runtime package instead of relying on a system wide copy: " +
                $"Microsoft.ML.OnnxRuntime (CPU), Microsoft.ML.OnnxRuntime.Gpu (CUDA and TensorRT), " +
                $"Microsoft.ML.OnnxRuntime.DirectML or Microsoft.ML.OnnxRuntime.QNN, all with version {version}. " +
                $"Probed '{directory}' and its 'runtimes/<rid>/native' folder. " +
                $"Note: on Windows a mismatched ONNX Runtime (for example the 1.17 build in C:\\Windows\\System32 that " +
                $"ships with Windows ML) is picked up from the system and crashes the process when the application " +
                $"does not carry its own copy.";
        }

        private static string ResolveNativeLibraryFileName()
        {
            if (OperatingSystem.IsWindows())
            {
                return "onnxruntime.dll";
            }

            return OperatingSystem.IsMacOS() ? "libonnxruntime.dylib" : "libonnxruntime.so";
        }
    }
}
