namespace SileroVad
{
    // Helper class to restore original speech timestamps.
    public class SpeechTimestampsMap
    {

        public List<int> chunk_end_sample;

        public int sampling_rate;

        public int time_precision;

        public List<int> total_silence_before;

        public SpeechTimestampsMap(List<Dictionary<string, int>> chunks, int sampling_rate, int time_precision = 2)
        {
            this.sampling_rate = sampling_rate;
            this.time_precision = time_precision;
            chunk_end_sample = new List<int>();
            total_silence_before = new List<int>();
            var previous_end = 0;
            var silent_samples = 0;
            foreach (var chunk in chunks)
            {
                silent_samples += chunk["start"] - previous_end;
                previous_end = chunk["end"];
                chunk_end_sample.Add(chunk["end"] - silent_samples);
                total_silence_before.Add(silent_samples / sampling_rate);
            }
        }
    }
}