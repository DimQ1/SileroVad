using SileroVad.Properties;

namespace SileroVad
{
    public class Vad: IDisposable
    {
        private readonly SileroVadModel _model;
        private bool disposedValue;

        public Vad()
        {

            this._model = new SileroVadModel(Resources.silero_vad);

        }

        // The code below is adapted from https://github.com/snakers4/silero-vad.
        // This method is used for splitting long audios into speech chunks using silero VAD.
        //     Args:
        //       audio: One dimensional float array.
        //       threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        //         probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        //         parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
        //       min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
        //       max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
        //         than max_speech_duration_s will be split at the timestamp of the last silence that
        //         lasts more than 100s (if any), to prevent agressive cutting. Otherwise, they will be
        //         split aggressively just before max_speech_duration_s.
        //       min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
        //         before separating it
        //       window_size_samples: Audio chunks of window_size_samples size are fed to the silero VAD model.
        //         WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate.
        //         Values other than these may affect model perfomance!!
        //       speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side
        //     Returns:
        //       List of dicts containing begin and end samples of each speech chunk.
        //     
        public List<VadSpeech> GetSpeechTimestamps(
            ReadOnlySpan<float> audio,
            float threshold = 0.5f,
            int min_speech_duration_ms = 50,
            float max_speech_duration_s = float.PositiveInfinity,
            int min_silence_duration_ms = 2000,
            int window_size_samples = 1024,
            int speech_pad_ms = 100)
        {
            if (!new List<int> { 512, 1024, 1536 }.Contains(window_size_samples))
            {
                Console.WriteLine("Unusual window_size_samples! Supported window_size_samples:\n - [512, 1024, 1536] for 16000 sampling_rate");
            }

            var batchSize = 1;
            var sampling_rate = 16000;
            var min_speech_samples = sampling_rate * min_speech_duration_ms / 1000;
            var speech_pad_samples = sampling_rate * speech_pad_ms / 1000;
            var max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples;
            var min_silence_samples = sampling_rate * min_silence_duration_ms / 1000;
            var min_silence_samples_at_max_speech = sampling_rate * 98 / 1000;
            var audio_length_samples = audio.Length;
            var (hTensor, cTensor, sampleRateTensor) = _model.GetInitialStateTensors(batchSize, sampling_rate);
            var state = (hTensor, cTensor);
            var speech_probs = new List<float>();

            foreach (var chunk in Enumerable.Range(0, audio.Length).Chunk(window_size_samples))
            {
                (var speech_prob, state) = _model.DetectSpeach(audio.Slice(chunk.First(), chunk.Length), state, sampleRateTensor, batchSize);
                speech_probs.Add(speech_prob);
            }

            var triggered = false;
            var speeches = new List<VadSpeech>();
            var current_speech = new VadSpeech();

            var neg_threshold = threshold - 0.15;
            // to save potential segment end (and tolerate some silence)
            var temp_end = 0;
            // to save potential segment limits in case of maximum segment size reached
            var prev_end = 0;
            var next_start = 0;

            for (int i = 0; i < speech_probs.Count; i++)
            {
                var speech_prob = speech_probs[i];

                if (speech_prob >= threshold && temp_end > 0)
                {
                    temp_end = 0;
                    if (next_start < prev_end)
                    {
                        next_start = window_size_samples * i;
                    }
                }
                if (speech_prob >= threshold && !triggered)
                {
                    triggered = true;
                    current_speech.Start = window_size_samples * i;
                    continue;
                }
                if (triggered && window_size_samples * i - current_speech.Start > max_speech_samples)
                {
                    if (prev_end > 0)
                    {
                        current_speech.End = prev_end;
                        speeches.Add(current_speech);
                        current_speech = new VadSpeech();
                        // previously reached silence (< neg_thres) and is still not speech (< thres)
                        if (next_start < prev_end)
                        {
                            triggered = false;
                        }
                        else
                        {
                            current_speech.Start = next_start;
                        }
                        prev_end = next_start = temp_end = 0;
                    }
                    else
                    {
                        current_speech.End = window_size_samples * i;
                        speeches.Add(current_speech);
                        current_speech = new VadSpeech();
                        prev_end = next_start = temp_end = 0;
                        triggered = false;
                        continue;
                    }
                }

                if (speech_prob < neg_threshold && triggered)
                {
                    if (temp_end == 0)
                    {
                        temp_end = window_size_samples * i;
                    }
                    // condition to avoid cutting in very short silence
                    if (window_size_samples * i - temp_end > min_silence_samples_at_max_speech)
                    {
                        prev_end = temp_end;
                    }

                    if (window_size_samples * i - temp_end >= min_silence_samples)
                    {
                        current_speech.End = temp_end;
                        if (current_speech.End - current_speech.Start > min_speech_samples)
                        {
                            speeches.Add(current_speech);
                        }
                        current_speech = new VadSpeech();
                        prev_end = next_start = temp_end = 0;
                        triggered = false;
                    }
                }
            }

            if (current_speech.Start > 0 && audio_length_samples - current_speech.Start > min_speech_samples)
            {
                current_speech.End = audio_length_samples;
                speeches.Add(current_speech);
            }

            foreach (var (i, speech) in speeches.Select((_p_3, _p_4) => Tuple.Create(_p_4, _p_3)))
            {
                if (i == 0)
                {
                    speech.Start = Convert.ToInt32(Math.Max(0, speech.Start - speech_pad_samples));
                }
                if (i != speeches.Count - 1)
                {
                    var silence_duration = speeches[i + 1].Start - speech.End;
                    if (silence_duration < 2 * speech_pad_samples)
                    {
                        speech.End += Convert.ToInt32(silence_duration / 2);
                        speeches[i + 1].Start = Convert.ToInt32(Math.Max(0, speeches[i + 1].Start - silence_duration / 2));
                    }
                    else
                    {
                        speech.End = Convert.ToInt32(Math.Min(audio_length_samples, speech.End + speech_pad_samples));
                        speeches[i + 1].Start = Convert.ToInt32(Math.Max(0, speeches[i + 1].Start - speech_pad_samples));
                    }
                }
                else
                {
                    speech.End = Convert.ToInt32(Math.Min(audio_length_samples, speech.End + speech_pad_samples));
                }
            }
            return speeches;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                   this._model.Dispose();
                }

                disposedValue = true;
            }
        }

        // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        ~Vad()
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