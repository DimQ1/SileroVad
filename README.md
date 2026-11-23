# Voice Activity Detection for .Net

[![NuGet](https://img.shields.io/nuget/v/SileroVad?label=NuGet&color=004880)](https://www.nuget.org/packages/SileroVad)
[![Downloads](https://img.shields.io/nuget/dt/SileroVad?label=downloads&color=orange)](https://www.nuget.org/packages/SileroVad)
[![Tests](https://github.com/DimQ1/SileroVad/actions/workflows/publish.yml/badge.svg?branch=master)](https://github.com/DimQ1/SileroVad/actions/workflows/publish.yml)
[![Release](https://img.shields.io/github/v/release/DimQ1/SileroVad?label=release&color=blueviolet)](https://github.com/DimQ1/SileroVad/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/DimQ1/SileroVad/blob/master/SileroVad/LICENSE.txt)
[![.NET](https://img.shields.io/badge/.NET-8.0%20%7C%209.0%20%7C%2010.0%20%7C%2011.0-512BD4)](https://dotnet.microsoft.com/download/dotnet/11.0)

Targets **.NET 8, 9, 10 and 11** (`net8.0`, `net9.0`, `net10.0`, `net11.0`); the lowest supported framework is `net8.0`.

Wrapper around the official [Silero VAD](https://github.com/snakers4/silero-vad) ONNX models.

## Models

| Revision | Bundled file | Recurrent state | Window at 16 kHz | Sample rates |
| --- | --- | --- | --- | --- |
| **v5** (default) – silero-vad **v6.3** model | `silero_vad.onnx` | `state` [2, N, 128] plus 64 samples of context | 512 | 8 kHz, 16 kHz |
| **v4** (legacy) | `silero_vad_v4.onnx` | `h`/`c` [2, N, 64] | 512 / 1024 / 1536 | 16 kHz |

`VadModelKind.V5` is the input/output layout introduced by silero-vad v5; the v6.x models keep it, so the bundled
v6.3 model and any v5/v6 model you load yourself are all `V5`. The revision is detected from the model's input names,
so every model is used through the same API:

```csharp
using var latest = new Vad();                              // bundled v6.3 model (v5 layout)
using var legacy = new Vad(VadModelKind.V4);               // bundled v4 model
using var custom = new Vad("path/to/silero_vad.onnx");     // your own model
```

Notes on the v6.3 model:

* Compared to v6.2 it finds short pauses inside speech more reliably, closes speech faster and gives fewer false
  positives on noise. At the same `Threshold` it is slightly stricter, so re-check a threshold tuned for v6.2.
* Upstream also ships [`silero_vad_16k_op15.onnx`](https://github.com/snakers4/silero-vad/tree/v6.3/src/silero_vad/data),
  a 16 kHz only export without `If` nodes. It gives the same probabilities at 16 kHz and may suit execution
  providers that handle control flow poorly (DirectML, QNN, TensorRT, WebGPU). Load it with
  `new Vad("silero_vad_16k_op15.onnx")` and only feed it 16 kHz audio (or a multiple of 16 kHz): it ignores the
  `sr` input, so 8 kHz audio would give wrong results.

> **Before you run:** your application has to reference **one** ONNX Runtime package — see
> [Runtime package](#runtime-package-required). The library itself only depends on the managed runtime.

## Example

A complete, runnable example lives in [**SileroVad.Samples**](https://github.com/DimQ1/SileroVad/tree/master/SileroVad.Samples): it reads an audio file with
[NAudio](https://github.com/naudio/NAudio), resamples it when the sample rate is not supported, detects the
speech and writes it to `<input>.speech.wav`.

```bash
dotnet run --project SileroVad.Samples
dotnet run --project SileroVad.Samples -- "C:\path\to\audio.wav"
```

See [`SileroVad.Samples/SpeechExtractor.cs`](https://github.com/DimQ1/SileroVad/blob/master/SileroVad.Samples/SpeechExtractor.cs) for the code and
[`SileroVad.Samples/README.md`](https://github.com/DimQ1/SileroVad/blob/master/SileroVad.Samples/README.md) for the details, including the NAudio 3.x
specifics. The historical overload still works as before:
`Vad.GetSpeechTimestamps(audio, min_silence_duration_ms: 500, threshold: 0.5f)`.

## Features

* **Batch detection** – `Vad.GetSpeechTimestamps(audio, new VadOptions { ... })` returns the speech chunks of a
  whole recording.
* **Speech probabilities** – `Vad.GetSpeechProbabilities(audio)` exposes the raw per-window model output, and
  `Vad.GetSpeechTimestampsFromProbabilities(probabilities, options, audio.Length)` turns it into speech chunks
  without running the model again (handy to tune `Threshold` and the durations on the same audio).
* **Streaming** – `Vad.CreateStream()` detects speech incrementally, as audio arrives:

  ```csharp
  using var vad = new Vad();
  using var stream = vad.CreateStream(new VadOptions { SampleRate = 16000 });

  // Feed arbitrary chunk sizes; completed segments are returned as soon as they close.
  foreach (var segment in stream.Push(samples)) { /* ... */ }
  foreach (var segment in stream.Flush()) { /* ... */ }        // end of stream
  ```

* **More sample rates** – 8 kHz, 16 kHz and any multiple of 16 kHz (audio is decimated to 16 kHz before
  inference and reported positions stay in the sample rate you pass in).
* **Helpers** – `VadHelper` extracts speech or silence samples, merges close segments, converts positions to
  `TimeSpan`s; `VadSpeech` exposes `Length`, `Duration`, `Contains` and `Overlaps`.
* **Model control** – `VadModel`/`VadState` give direct access to the model and to the recurrent state for
  custom pipelines and batches.

`VadOptions` carries the tunables (`Threshold`, `NegThreshold`, `MinSpeechDurationMs`, `MinSilenceDurationMs`,
`MaxSpeechDurationSeconds`, `SpeechPadMs`, `WindowSizeSamples`, ...). `NegThreshold` is the exit threshold that
ends speech; it defaults to `max(Threshold - 0.15, 0.01)`, like the reference implementation. Its defaults match the historical
`GetSpeechTimestamps` overload, so switching between the two does not change results.

## Runtime package (required)

`SileroVad` references **only** `Microsoft.ML.OnnxRuntime.Managed`. That is deliberate: the application then
picks **exactly one** native ONNX Runtime, so two packages never fight over the same `onnxruntime` binary, and
the execution provider (CPU, CUDA, DirectML, QNN, a plugin, ...) stays a deployment decision instead of a
library constraint.

Add one runtime package to the application, with the **same version as the managed package** (1.30.0):

| Provider | Package |
| --- | --- |
| CPU – works everywhere | `Microsoft.ML.OnnxRuntime` |
| CUDA and TensorRT (Windows/Linux) | `Microsoft.ML.OnnxRuntime.Gpu` (or `.Gpu.Windows` / `.Gpu.Linux`) |
| DirectML (Windows) | `Microsoft.ML.OnnxRuntime.DirectML` |
| Qualcomm NPU | `Microsoft.ML.OnnxRuntime.QNN` |
| WebGPU and Windows ML / Foundry | `Microsoft.ML.OnnxRuntime.EP.WebGpu`, `Microsoft.ML.OnnxRuntime.Foundry` (plugins) |

```xml
<ItemGroup>
  <!-- one, and only one, runtime package -->
  <PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.30.0" />
</ItemGroup>
```

### Platform support

The library is pure managed code and ships plain `net8.0` / `net9.0` / `net10.0` / `net11.0` assets, which are exactly what
`net8.0-android`, `net9.0-ios`, `net10.0-windows` and the other platform specific target frameworks consume
(NuGet picks the best asset of every package separately), so no platform specific target is published.

The native ONNX Runtime, however, has to exist for the platform you deploy to:

| Platform | Native runtime |
| --- | --- |
| Windows, Linux, macOS | one of the packages above |
| Android | no official .NET package: bring `libonnxruntime.so` yourself (a third party runtime package such as `NtvLibs.OnnxRuntime.CPU.runtime.android`, or your own build of the ONNX Runtime Android binaries) |
| iOS, MacCatalyst | no official .NET package: link/build ONNX Runtime yourself |

ONNX Runtime's managed package only contains the interop for the mobile target frameworks, not the native
libraries, so a platform specific target framework on this library would not make anything runnable by itself.

## Choosing the execution provider

Providers are requested through the constructor overloads that take an `Action<SessionOptions>`. A request for
a provider that the loaded runtime does not offer is **ignored by default**, so the same configuration runs on
a GPU machine and on a laptop. Short names (`cuda`, `dml`, `qnn`, ...) are accepted as well, and the helper
picks the right ONNX Runtime call for each provider — CUDA, TensorRT, ROCm, MIGraphX and DirectML must not go
through the generic name based API, which rejects them:

```csharp
using SileroVad;

// CUDA when present, otherwise the next provider (CPU last)
using var vad = new Vad(VadModelKind.V5, VadExecutionProviders.Use(
    VadExecutionProviders.Cuda,
    new Dictionary<string, string> { ["device_id"] = "0" }));

// DirectML on Windows
using var dml = new Vad(VadModelKind.V5, VadExecutionProviders.Use(VadExecutionProviders.DirectMl));

// Everything SessionOptions offers, including plugin providers (WebGPU, Windows ML, ...)
using var tuned = new Vad(VadModelKind.V5, options =>
{
    OrtEnv.Instance().RegisterExecutionProviderLibrary("webgpu", "path/to/webgpu_provider.dll");
    options.AppendExecutionProvider(VadExecutionProviders.WebGpu, new Dictionary<string, string>());
    options.IntraOpNumThreads = 2;
    options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
});

// Fail fast instead of silently running on CPU
using var strict = new Vad(
    VadModelKind.V5,
    VadExecutionProviders.Use(VadExecutionProviders.Cuda, throwIfUnavailable: true));
```

### Diagnostics

```csharp
VadRuntime.EnsureNativeRuntimeAvailable();          // throws with instructions instead of crashing
VadRuntime.ResolveNativeLibraryPath();              // which native library the application carries
VadRuntime.GetAvailableExecutionProviders();        // e.g. CPUExecutionProvider, CUDAExecutionProvider
VadExecutionProviders.IsAvailable(VadExecutionProviders.Cuda);
```

`VadModel` and `Vad` call `EnsureNativeRuntimeAvailable()` themselves, so a missing or mismatched runtime is
reported before ONNX Runtime is touched.

### Why conflicts happen, and what to do

1. **Two runtime packages in one application.** `Microsoft.ML.OnnxRuntime` and `Microsoft.ML.OnnxRuntime.Gpu`
   both ship `onnxruntime` and overwrite each other; which copy wins depends on resolution order. Reference
   one of them.
2. **The operating system supplies its own copy.** Windows ships `C:\Windows\System32\onnxruntime.dll`
   (1.17, from Windows ML). If the application does not carry a native runtime, that copy is loaded, its
   exports do not match the managed 1.30 API and the process dies with `0xC0000005` — **no catchable
   exception**. The application directory is searched before the system directories, so a proper runtime
   package wins; `VadRuntime.EnsureNativeRuntimeAvailable()` catches the broken case up front.
3. **Managed and native versions must match.** NuGet resolves `Microsoft.ML.OnnxRuntime.Managed` upward, so a
   provider package that lags behind (`.DirectML` and `.QNN` are still at 1.24.4) needs the managed package
   pinned down explicitly:

   ```xml
   <PackageReference Include="SileroVad" Version="1.3.0" />
   <PackageReference Include="Microsoft.ML.OnnxRuntime.DirectML" Version="1.24.4" />
   <PackageReference Include="Microsoft.ML.OnnxRuntime.Managed" Version="1.24.4" />
   ```

4. **Other components in the same process.** ML.NET (`Microsoft.ML.OnnxTransformer`), Whisper.net, Windows ML
   and friends also load ONNX Runtime. A process can only have one native `onnxruntime` loaded, so align the
   versions of all of them on the highest one in use.
5. **Provider binaries must come from the same package.** `onnxruntime_providers_cuda.dll`,
   `onnxruntime_providers_tensorrt.dll`, OpenVINO and plugin libraries are version-locked to the core
   library; do not mix a 1.24 provider with a 1.30 core.
6. **Single-file or trimmed publish.** Add `PublishSingleFile` with
   `<IncludeNativeLibrariesForSelfExtract>true</IncludeNativeLibrariesForSelfExtract>`, otherwise the native
   runtime and the provider libraries are not extracted next to the application.

## Backward compatibility

The historical API is unchanged and keeps working:

* `Vad()` – now uses the bundled **v5** model (previously v4).
* `Vad.GetSpeechTimestamps(audio, threshold, min_speech_duration_ms, max_speech_duration_s, min_silence_duration_ms, window_size_samples, speech_pad_ms)`
  – same signature, same 16 kHz semantics and the same default parameters. For v5 models the window size is
  fixed by the model (512 samples), so a passed window is only honoured for v4 models.
* `SileroVadModel`, including `GetInitialStateTensors` and `DetectSpeech`, plus `VadSpeech` and
  `VadHelper.GetSpeechSamples` keep their original signatures.
* `new Vad(VadModelKind.V4)` restores the previous behaviour exactly, including the previous window sizes.

Changes in 2.2.0 (silero-vad v6.3):

* The bundled `silero_vad.onnx` is the v6.3 model (it was v6.2). Inputs and outputs are unchanged.
* `MaxSpeechDurationSeconds`: when a chunk is cut with no silence to split at and speech goes on, the next chunk
  now starts at the cut. Previously the window at the cut was dropped.
* `VadStream`: with a large `SpeechPadMs` the left pad no longer reaches back into the previous segment, so
  streamed segments never overlap.
* New `VadOptions.NegThreshold`, `Vad.GetSpeechTimestampsFromProbabilities` and the `net11.0` target.

Fixed along the way: a segment that was still open at the end of the audio was dropped when it started in the
very first window (`current_speech.Start > 0` instead of "a segment is open" in the reference implementation).
Audio that begins with speech now reports that segment, which is most visible with the 512 sample windows of
the v5 model on short clips.

## Building and testing

```bash
dotnet build --configuration Release
dotnet test  --configuration Release
```

Building needs the .NET 11 SDK (see `global.json`). The test suite (`SileroVad.Tests`) runs on `net8.0`, `net9.0`,
`net10.0` and `net11.0` and checks both model revisions
against real speech audio: detection quality, silence handling, sample rates, streaming/batch parity and the
legacy API surface.

## References

<a id="1">[1]</a>
Silero Team. (2021).
Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier.
GitHub, GitHub repository, https://github.com/snakers4/silero-vad, hello@silero.ai.