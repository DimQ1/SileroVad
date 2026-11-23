using Xunit;

namespace SileroVad.Tests
{
    /// <summary>
    /// Model layout detection, checked against tiny synthetic models built with <see cref="OnnxModelBuilder"/>.
    /// </summary>
    public class VadModelLayoutTests
    {
        [Fact]
        public void ModelWithoutSileroInputs_IsRejected()
        {
            var model = new OnnxModelBuilder()
                .Input("x", OnnxModelBuilder.Float, "N", "S")
                .Identity("y", "x")
                .Build();

            var exception = Assert.Throws<NotSupportedException>(() => new VadModel(model));

            Assert.Contains("does not look like a Silero VAD model", exception.Message, StringComparison.Ordinal);
            Assert.Contains("x", exception.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void ModelWithoutStateInputs_IsRejected()
        {
            var model = new OnnxModelBuilder()
                .Input("input", OnnxModelBuilder.Float, "N", "S")
                .Input("sr", OnnxModelBuilder.Int64)
                .Identity("output", "input")
                .Build();

            var exception = Assert.Throws<NotSupportedException>(() => new VadModel(model));

            Assert.Contains("Unsupported Silero VAD model revision", exception.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void ModelWithBothStateLayouts_IsRejected()
        {
            var model = new OnnxModelBuilder()
                .Input("input", OnnxModelBuilder.Float, "N", "S")
                .Input("sr", OnnxModelBuilder.Int64)
                .Input("state", OnnxModelBuilder.Float, "2", "N", "128")
                .Input("h", OnnxModelBuilder.Float, "2", "N", "64")
                .Input("c", OnnxModelBuilder.Float, "2", "N", "64")
                .Identity("output", "input")
                .Build();

            var exception = Assert.Throws<NotSupportedException>(() => new VadModel(model));

            Assert.Contains("both v4 and v5", exception.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void ModelWithoutProbabilityOutput_IsRejected()
        {
            var model = new OnnxModelBuilder()
                .Input("input", OnnxModelBuilder.Float, "N", "S")
                .Input("sr", OnnxModelBuilder.Int64)
                .Input("state", OnnxModelBuilder.Float, "2", "N", "128")
                .Identity("probability", "input")
                .Build();

            var exception = Assert.Throws<NotSupportedException>(() => new VadModel(model));

            Assert.Contains("'output'", exception.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void SymbolicStateWidth_FallsBackToTheRevisionDefault()
        {
            var v5 = new OnnxModelBuilder()
                .Input("input", OnnxModelBuilder.Float, "N", "S")
                .Input("sr", OnnxModelBuilder.Int64)
                .Input("state", OnnxModelBuilder.Float, "2", "N", "W")
                .Identity("output", "input")
                .Identity("stateN", "state")
                .Build();

            var v4 = new OnnxModelBuilder()
                .Input("input", OnnxModelBuilder.Float, "N", "S")
                .Input("sr", OnnxModelBuilder.Int64)
                .Input("h", OnnxModelBuilder.Float, "2", "N", "W")
                .Input("c", OnnxModelBuilder.Float, "2", "N", "W")
                .Identity("output", "input")
                .Identity("hn", "h")
                .Identity("cn", "c")
                .Build();

            using var v5Model = new VadModel(v5);
            using var v4Model = new VadModel(v4);

            Assert.Equal(VadModelKind.V5, v5Model.Kind);
            Assert.Equal(128, v5Model.StateSize);
            Assert.Equal(VadModelKind.V4, v4Model.Kind);
            Assert.Equal(64, v4Model.StateSize);
        }

        [Fact]
        public void V5_PrependsTheTailOfThePreviousWindowAsContext()
        {
            // An identity model returns its own input, so the "probability" is the first sample the model saw:
            // zero context on the first call, then sample 512 - 64 of the previous window.
            var model = new OnnxModelBuilder()
                .Input("input", OnnxModelBuilder.Float, "N", "S")
                .Input("sr", OnnxModelBuilder.Int64)
                .Input("state", OnnxModelBuilder.Float, "2", "N", "128")
                .Identity("output", "input")
                .Identity("stateN", "state")
                .Build();

            using var vadModel = new VadModel(model);
            var state = vadModel.CreateState(TestAudio.SampleRate);
            var first = Enumerable.Range(1, 512).Select(i => (float)i).ToArray();
            var second = Enumerable.Range(1001, 512).Select(i => (float)i).ToArray();

            Assert.Equal(0f, vadModel.DetectSpeech(first, state, TestAudio.SampleRate));
            Assert.Equal(first[512 - 64], vadModel.DetectSpeech(second, state, TestAudio.SampleRate));

            state.Reset();

            Assert.Equal(0f, vadModel.DetectSpeech(second, state, TestAudio.SampleRate));
        }

        [Fact]
        public void V4_FeedsTheWindowWithoutContext()
        {
            var model = new OnnxModelBuilder()
                .Input("input", OnnxModelBuilder.Float, "N", "S")
                .Input("sr", OnnxModelBuilder.Int64)
                .Input("h", OnnxModelBuilder.Float, "2", "N", "64")
                .Input("c", OnnxModelBuilder.Float, "2", "N", "64")
                .Identity("output", "input")
                .Identity("hn", "h")
                .Identity("cn", "c")
                .Build();

            using var vadModel = new VadModel(model);
            var state = vadModel.CreateState(TestAudio.SampleRate);
            var window = Enumerable.Range(7, 1024).Select(i => (float)i).ToArray();

            Assert.Equal(7f, vadModel.DetectSpeech(window, state, TestAudio.SampleRate));
            Assert.Equal(7f, vadModel.DetectSpeech(window, state, TestAudio.SampleRate));
        }
    }
}
