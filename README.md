# Voice Activity Detection for .Net

### Quick Start

```html
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using SileroVad;

namespace WhisperONNX
{
    public static class FileReader
    {
        private static int SAMPLE_RATE = 16000;

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

            List<VadSpeech> resul = Vad.GetSpeechTimestamps(array, min_silence_duration_ms: 500, threshold: 0.4f);

            var audioSpeech = Vad.GetSpeechSamples(array, resul);

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
}
```


## References

<a id="1">[1]</a>
Silero Team. (2021).
Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier.
GitHub, GitHub repository, https://github.com/snakers4/silero-vad, hello@silero.ai.