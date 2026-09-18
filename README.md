# Voice Activity Detection for .Net

Targets **.NET 10** (`net10.0`).

Wrapper around the official [Silero VAD](https://github.com/snakers4/silero-vad) ONNX models.

## Models

| Revision | Bundled file | Recurrent state | Window at 16 kHz | Sample rates |
| --- | --- | --- | --- | --- |
| **v5** (default) | `silero_vad.onnx` | `state` [2, N, 128] plus 64 samples of context | 512 | 8 kHz, 16 kHz |
| **v4** (legacy) | `silero_vad_v4.onnx` | `h`/`c` [2, N, 64] | 512 / 1024 / 1536 | 16 kHz |

The revision is detected from the model's input names, so both are used through the same API:

```csharp
using var latest = new Vad();                              // bundled v5 model
using var legacy = new Vad(VadModelKind.V4);               // bundled v4 model
using var custom = new Vad("path/to/silero_vad.onnx");     // your own model
```

### Quick Start

```html
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using SileroVad;

 public static class FileReader
    {
        private static int SAMPLE_RATE = 16000;
        private static Vad vad = new Vad();

        public static void VadFile(string filePath)
        {
            var ext = Path.GetExtension(filePath).ToLower();
            WaveStream waveFileReader;

            switch (ext)
            {
                case ".wav":
                    waveFileReader = new WaveFileReader(filePath);
                    break;
                case ".mp3":
                    waveFileReader = new Mp3FileReader(filePath);
                    break;
                default:
                    throw new NotSupportedException($"not supported {ext}");
            }

            var TotalTime = waveFileReader.TotalTime;

            ISampleProvider sampleProvider;

            if (waveFileReader.WaveFormat.SampleRate != SAMPLE_RATE)
            {
                sampleProvider = new WdlResamplingSampleProvider(waveFileReader.ToSampleProvider(), SAMPLE_RATE).ToMono();
            }
            else
            {
                sampleProvider = waveFileReader.ToSampleProvider();
            }

            var array = new float[CountSamples(TotalTime)];

            sampleProvider.Read(array, 0, array.Length);

            List<VadSpeech> resul = vad.GetSpeechTimestamps(array, min_silence_duration_ms: 500, threshold: 0.5f);

            var audioSpeech = VadHelper.GetSpeechSamples(array, resul);

            var fileTrim = Path.ChangeExtension(filePath, "speech") + ".wav";

            using var fileWriter = new WaveFileWriter(fileTrim, new WaveFormat(16000, 1));
            foreach (var sample in audioSpeech)
            {
                fileWriter.WriteSample(sample);
            }
            fileWriter.Flush();
            waveFileReader.Dispose();

        }

        private static int CountSamples(TimeSpan time)
        {
            WaveFormat waveFormat = new WaveFormat(16000, 1);

            return TimeSpanToSamples(time, waveFormat);
        }

        private static int TimeSpanToSamples(TimeSpan time, WaveFormat waveFormat)
        {
            return (int)(time.TotalSeconds * (double)waveFormat.SampleRate) * waveFormat.Channels;
        }
    }
```


## Features

* **Batch detection** – `Vad.GetSpeechTimestamps(audio, new VadOptions { ... })` returns the speech chunks of a
  whole recording.
* **Speech probabilities** – `Vad.GetSpeechProbabilities(audio)` exposes the raw per-window model output.
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

`VadOptions` carries the tunables (`Threshold`, `MinSpeechDurationMs`, `MinSilenceDurationMs`,
`MaxSpeechDurationSeconds`, `SpeechPadMs`, `WindowSizeSamples`, ...). Its defaults match the historical
`GetSpeechTimestamps` overload, so switching between the two does not change results.

## ONNX Runtime

The library references `Microsoft.ML.OnnxRuntime` (CPU), so it has no CUDA dependency and runs anywhere.
To run inference on a CUDA capable GPU, replace that package reference in your application with
`Microsoft.ML.OnnxRuntime.Gpu` of the same version — both ship the same managed API and the library needs no
change. (Verified: the CPU and GPU providers return identical probabilities for these models.)

## Backward compatibility

The historical API is unchanged and keeps working:

* `Vad()` – now uses the bundled **v5** model (previously v4).
* `Vad.GetSpeechTimestamps(audio, threshold, min_speech_duration_ms, max_speech_duration_s, min_silence_duration_ms, window_size_samples, speech_pad_ms)`
  – same signature, same 16 kHz semantics and the same default parameters. For v5 models the window size is
  fixed by the model (512 samples), so a passed window is only honoured for v4 models.
* `SileroVadModel`, including `GetInitialStateTensors` and `DetectSpeech`, plus `VadSpeech` and
  `VadHelper.GetSpeechSamples` keep their original signatures.
* `new Vad(VadModelKind.V4)` restores the previous behaviour exactly, including the previous window sizes.

Fixed along the way: a segment that was still open at the end of the audio was dropped when it started in the
very first window (`current_speech.Start > 0` instead of "a segment is open" in the reference implementation).
Audio that begins with speech now reports that segment, which is most visible with the 512 sample windows of
the v5 model on short clips.

## Building and testing

```bash
dotnet build --configuration Release
dotnet test  --configuration Release
```

The test suite (`SileroVad.Tests`) checks both model revisions against real speech audio: detection quality,
silence handling, sample rates, streaming/batch parity and the legacy API surface.

## References

<a id="1">[1]</a>
Silero Team. (2021).
Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier.
GitHub, GitHub repository, https://github.com/snakers4/silero-vad, hello@silero.ai.