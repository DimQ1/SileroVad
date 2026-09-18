namespace SileroVad
{
    /// <summary>
    /// Silero VAD ONNX model revisions understood by the library.
    /// </summary>
    /// <remarks>
    /// The revision is detected automatically from the model's input tensor names, so the same code
    /// works with both families of models.
    /// </remarks>
    public enum VadModelKind
    {
        /// <summary>
        /// Legacy v4 models: <c>input</c>/<c>sr</c>/<c>h</c>/<c>c</c> inputs and <c>output</c>/<c>hn</c>/<c>cn</c> outputs.
        /// The caller feeds raw audio windows without context and carries two 64-wide recurrent state tensors.
        /// </summary>
        V4 = 4,

        /// <summary>
        /// Current v5 models: <c>input</c>/<c>state</c>/<c>sr</c> inputs and <c>output</c>/<c>stateN</c> outputs.
        /// The caller prepends the previous 64 samples (32 at 8 kHz) of context to every window and carries
        /// a single 128-wide state tensor.
        /// </summary>
        V5 = 5,
    }
}
