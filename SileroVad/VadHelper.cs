namespace SileroVad
{
    public static class VadHelper
    {
        public static IEnumerable<float> GetSpeechSamples(float[] audio, List<VadSpeech> vadSpeeches)
        {
            return vadSpeeches.SelectMany(speech => audio[speech.Start..speech.End]);
        }
    }
}
