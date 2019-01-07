using System.Collections.Generic;

namespace Microsoft.ML.Auto.Public
{
    public class PipelineElement
    {
        public readonly string Name;
        public readonly PipelineElementType ElementType;
        public readonly string[] InColumns;
        public readonly string[] OutColumns;
        public readonly IDictionary<string, object> Properties;

        public PipelineElement(string name, PipelineElementType elementType,
            string[] inColumns, string[] outColumns,
            IDictionary<string, object> properties)
        {
            Name = name;
            ElementType = elementType;
            InColumns = inColumns;
            OutColumns = outColumns;
            Properties = properties;
        }
    }
}
