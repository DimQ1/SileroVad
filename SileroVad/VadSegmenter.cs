namespace SileroVad
{
    /// <summary>
    /// Turns a stream of per-window speech probabilities into speech segments.
    /// </summary>
    /// <remarks>
    /// This is a port of the splitting logic used by the reference implementation
    /// (<c>get_speech_timestamps</c> in <see href="https://github.com/snakers4/silero-vad">silero-vad</see>),
    /// shared by the batch (<see cref="Vad"/>) and streaming (<see cref="VadStream"/>) APIs.
    /// </remarks>
    internal sealed class VadSegmenter
    {
        private readonly int _windowSizeSamples;
        private readonly float _threshold;
        private readonly float _negThreshold;
        private readonly int _minSpeechSamples;
        private readonly float _maxSpeechSamples;
        private readonly int _minSilenceSamples;
        private readonly int _minSilenceSamplesAtMaxSpeech;
        private readonly bool _useLongestSilenceForMaxSpeech;
        private readonly List<(int End, int Duration)> _possibleEnds = new();

        private bool _triggered;
        private bool _segmentOpen;
        private VadSpeech _current = new();
        private int _tempEnd;
        private int _prevEnd;
        private int _nextStart;

        public VadSegmenter(int windowSizeSamples, int sampleRate, VadOptions options)
        {
            _windowSizeSamples = windowSizeSamples;
            _threshold = options.Threshold;
            _negThreshold = Math.Max(options.Threshold - 0.15f, 0.01f);
            _minSpeechSamples = sampleRate * options.MinSpeechDurationMs / 1000;
            SpeechPadSamples = sampleRate * options.SpeechPadMs / 1000;
            _maxSpeechSamples = sampleRate * options.MaxSpeechDurationSeconds - windowSizeSamples - 2 * SpeechPadSamples;
            _minSilenceSamples = sampleRate * options.MinSilenceDurationMs / 1000;
            _minSilenceSamplesAtMaxSpeech = sampleRate * options.MinSilenceAtMaxSpeechMs / 1000;
            _useLongestSilenceForMaxSpeech = options.UseLongestSilenceForMaxSpeech;
        }

        /// <summary>Samples of padding applied to both sides of a segment.</summary>
        public int SpeechPadSamples { get; }

        /// <summary>
        /// Feeds the probability of one window. Segments that finished with this window are appended to
        /// <paramref name="closedSegments"/> without padding.
        /// </summary>
        /// <param name="speechProbability">Probability returned by the model for the window.</param>
        /// <param name="windowIndex">Zero based index of the window inside the audio.</param>
        /// <param name="closedSegments">Receives the segments closed by this window.</param>
        public void Append(float speechProbability, int windowIndex, List<VadSpeech> closedSegments)
        {
            var position = _windowSizeSamples * windowIndex;

            // Speech returned after a temporary end: remember the silence as a possible split point.
            if (speechProbability >= _threshold && _tempEnd > 0)
            {
                var silenceDuration = position - _tempEnd;
                if (_useLongestSilenceForMaxSpeech && silenceDuration > _minSilenceSamplesAtMaxSpeech)
                {
                    _possibleEnds.Add((_tempEnd, silenceDuration));
                }

                _tempEnd = 0;
                if (_nextStart < _prevEnd)
                {
                    _nextStart = position;
                }
            }

            // Start of speech.
            if (speechProbability >= _threshold && !_triggered)
            {
                _triggered = true;
                _segmentOpen = true;
                _current.Start = position;
                return;
            }

            // Maximum speech length reached: decide where to cut.
            if (_triggered && position - _current.Start > _maxSpeechSamples)
            {
                if (_useLongestSilenceForMaxSpeech && _possibleEnds.Count > 0)
                {
                    var (prevEnd, duration) = LongestPossibleEnd();

                    _current.End = prevEnd;
                    closedSegments.Add(_current);
                    _current = new VadSpeech();
                    _segmentOpen = false;

                    var nextStart = prevEnd + duration;
                    if (nextStart < prevEnd + position)
                    {
                        _current.Start = nextStart;
                        _segmentOpen = true;
                    }
                    else
                    {
                        _triggered = false;
                    }

                    _prevEnd = _nextStart = _tempEnd = 0;
                    _possibleEnds.Clear();
                }
                else if (_prevEnd > 0)
                {
                    _current.End = _prevEnd;
                    closedSegments.Add(_current);
                    _current = new VadSpeech();

                    // Previously reached silence (< neg_threshold) and is still not speech (< threshold).
                    if (_nextStart < _prevEnd)
                    {
                        _triggered = false;
                        _segmentOpen = false;
                    }
                    else
                    {
                        _current.Start = _nextStart;
                    }

                    _prevEnd = _nextStart = _tempEnd = 0;
                    _possibleEnds.Clear();
                }
                else
                {
                    _current.End = position;
                    closedSegments.Add(_current);
                    _current = new VadSpeech();
                    _prevEnd = _nextStart = _tempEnd = 0;
                    _triggered = false;
                    _segmentOpen = false;
                    _possibleEnds.Clear();
                    return;
                }
            }

            // Silence detection while in speech.
            if (speechProbability < _negThreshold && _triggered)
            {
                if (_tempEnd == 0)
                {
                    _tempEnd = position;
                }

                // Condition to avoid cutting in very short silence.
                if (!_useLongestSilenceForMaxSpeech && position - _tempEnd > _minSilenceSamplesAtMaxSpeech)
                {
                    _prevEnd = _tempEnd;
                }

                if (position - _tempEnd >= _minSilenceSamples)
                {
                    _current.End = _tempEnd;
                    if (_current.End - _current.Start > _minSpeechSamples)
                    {
                        closedSegments.Add(_current);
                    }

                    _current = new VadSpeech();
                    _prevEnd = _nextStart = _tempEnd = 0;
                    _triggered = false;
                    _segmentOpen = false;
                    _possibleEnds.Clear();
                }
            }
        }

        /// <summary>
        /// Ends the audio: a still open segment is closed at <paramref name="audioLengthSamples"/> when it is
        /// long enough.
        /// </summary>
        /// <param name="audioLengthSamples">Number of samples of the audio, padding excluded.</param>
        /// <param name="closedSegments">Receives the segments closed by the end of the audio.</param>
        public void Complete(int audioLengthSamples, List<VadSpeech> closedSegments)
        {
            // Upstream closes the trailing segment whenever one is still open, including one that starts at
            // sample 0; requiring Start > 0 used to drop speech that begins in the very first window.
            if (_segmentOpen && audioLengthSamples - _current.Start > _minSpeechSamples)
            {
                _current.End = audioLengthSamples;
                closedSegments.Add(_current);
                _current = new VadSpeech();
            }

            _triggered = false;
            _segmentOpen = false;
            _prevEnd = _nextStart = _tempEnd = 0;
            _possibleEnds.Clear();
        }

        /// <summary>
        /// Pads the segments by the configured amount and shares the remaining silence between neighbours that
        /// are closer than twice the padding.
        /// </summary>
        public static void ApplySpeechPadding(List<VadSpeech> speeches, int audioLengthSamples, int speechPadSamples)
        {
            for (var i = 0; i < speeches.Count; i++)
            {
                var speech = speeches[i];

                if (i == 0)
                {
                    speech.Start = Math.Max(0, speech.Start - speechPadSamples);
                }

                if (i != speeches.Count - 1)
                {
                    var silenceDuration = speeches[i + 1].Start - speech.End;
                    if (silenceDuration < 2 * speechPadSamples)
                    {
                        speech.End += silenceDuration / 2;
                        speeches[i + 1].Start = Math.Max(0, speeches[i + 1].Start - silenceDuration / 2);
                    }
                    else
                    {
                        speech.End = Math.Min(audioLengthSamples, speech.End + speechPadSamples);
                        speeches[i + 1].Start = Math.Max(0, speeches[i + 1].Start - speechPadSamples);
                    }
                }
                else
                {
                    speech.End = Math.Min(audioLengthSamples, speech.End + speechPadSamples);
                }
            }
        }

        private (int End, int Duration) LongestPossibleEnd()
        {
            var longest = _possibleEnds[0];

            foreach (var candidate in _possibleEnds)
            {
                if (candidate.Duration > longest.Duration)
                {
                    longest = candidate;
                }
            }

            return longest;
        }
    }
}
