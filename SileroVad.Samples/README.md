# SileroVad.Samples

Runnable example for the [SileroVad](../README.md) library: it reads an audio file, runs voice activity
detection over it and writes the speech it found to a new wav file.

```bash
dotnet run --project SileroVad.Samples
dotnet run --project SileroVad.Samples -- "C:\path\to\audio.wav"
```

Without arguments it analyses the 4 second speech excerpt that the test suite also uses, so the sample works
out of the box.

## What it shows

* `SpeechExtractor` – reading a file with [NAudio](https://github.com/naudio/NAudio), resampling when the
  sample rate is not one of the rates the model supports, and calling
  `Vad.GetSpeechTimestamps(audio, new VadOptions { ... })`.
* `VadHelper.GetSpeechSamples` – concatenating the detected speech into a single stream that is written back
  with `WaveFileWriter`.
* Reusing a single long-lived `Vad` instance: the detector keeps no state between calls, so one instance serves
  the whole application.
* The runtime package reference in `SileroVad.Samples.csproj`: the application, not the library, chooses the
  ONNX Runtime build and therefore the execution provider (see
  [Runtime package](../README.md#runtime-package-required)).

## Notes

* `VadOptions.SampleRate` must describe the audio you pass in. The detector accepts 8000, 16000 and multiples
  of 16000 Hz; `SileroVad.Samples` resamples everything else to 16 kHz with NAudio.
* mp3: this sample targets `net10.0` and handles `.wav`. `Mp3FileReader` lives in `NAudio.WinMM` on NAudio 3.x,
  so supporting mp3 needs a Windows target framework (`net10.0-windows`) and the `NAudio` metapackage.
* Output is written next to the input file as `<name>.speech.wav`.
