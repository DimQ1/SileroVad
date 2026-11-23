using System.Text;

namespace SileroVad.Tests
{
    /// <summary>
    /// Writes tiny ONNX models made of <c>Identity</c> nodes, to check how <see cref="VadModel"/> handles model
    /// layouts that are not (quite) Silero VAD without committing binary fixtures.
    /// </summary>
    /// <remarks>
    /// Only the few protobuf fields of <c>onnx.proto</c> that such a model needs are emitted: the IR version, the
    /// default opset, graph inputs with a symbolic shape, <c>Identity</c> nodes and untyped-shape outputs.
    /// </remarks>
    internal sealed class OnnxModelBuilder
    {
        internal const int Float = 1;
        internal const int Int64 = 7;

        private readonly List<(string Name, int ElementType, string[] Shape)> _inputs = new();
        private readonly List<(string Output, string Input)> _identities = new();

        /// <summary>Declares a graph input; dimensions are symbolic (<c>dim_param</c>) unless they parse as numbers.</summary>
        internal OnnxModelBuilder Input(string name, int elementType = Float, params string[] shape)
        {
            _inputs.Add((name, elementType, shape));
            return this;
        }

        /// <summary>Declares a graph output produced by an <c>Identity</c> node that copies <paramref name="input"/>.</summary>
        internal OnnxModelBuilder Identity(string output, string input)
        {
            _identities.Add((output, input));
            return this;
        }

        internal byte[] Build()
        {
            var graph = new ProtoWriter();
            foreach (var (output, input) in _identities)
            {
                graph.Message(1, node => node.String(1, input).String(2, output).String(4, "Identity"));
            }

            graph.String(2, "test");

            foreach (var (name, elementType, shape) in _inputs)
            {
                graph.Message(11, value => value.String(1, name).Message(2, type => type.Message(1, tensor =>
                {
                    tensor.Varint(1, elementType);
                    tensor.Message(2, tensorShape =>
                    {
                        foreach (var dimension in shape)
                        {
                            tensorShape.Message(1, dim =>
                            {
                                if (long.TryParse(dimension, out var size))
                                {
                                    dim.Varint(1, size);
                                }
                                else
                                {
                                    dim.String(2, dimension);
                                }
                            });
                        }
                    });
                })));
            }

            foreach (var (output, input) in _identities)
            {
                var elementType = _inputs.Single(declared => declared.Name == input).ElementType;
                graph.Message(12, value => value.String(1, output).Message(2, type => type.Message(1, tensor => tensor.Varint(1, elementType))));
            }

            return new ProtoWriter()
                .Varint(1, 8)                                            // ir_version
                .Message(8, opset => opset.String(1, string.Empty).Varint(2, 16)) // opset_import
                .Message(7, graph)                                       // graph
                .ToArray();
        }

        /// <summary>Minimal protobuf encoder: varints and length-delimited fields.</summary>
        private sealed class ProtoWriter
        {
            private readonly MemoryStream _stream = new();

            internal ProtoWriter Varint(int field, long value)
            {
                WriteTag(field, 0);
                WriteVarint((ulong)value);
                return this;
            }

            internal ProtoWriter String(int field, string value) => Bytes(field, Encoding.UTF8.GetBytes(value));

            internal ProtoWriter Message(int field, Action<ProtoWriter> build)
            {
                var nested = new ProtoWriter();
                build(nested);
                return Message(field, nested);
            }

            internal ProtoWriter Message(int field, ProtoWriter nested) => Bytes(field, nested.ToArray());

            internal byte[] ToArray() => _stream.ToArray();

            private ProtoWriter Bytes(int field, byte[] value)
            {
                WriteTag(field, 2);
                WriteVarint((ulong)value.Length);
                _stream.Write(value);
                return this;
            }

            private void WriteTag(int field, int wireType) => WriteVarint((ulong)((field << 3) | wireType));

            private void WriteVarint(ulong value)
            {
                while (value >= 0x80)
                {
                    _stream.WriteByte((byte)(value | 0x80));
                    value >>= 7;
                }

                _stream.WriteByte((byte)value);
            }
        }
    }
}
