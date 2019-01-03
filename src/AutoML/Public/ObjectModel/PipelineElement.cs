using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Auto.Public
{
    public class PipelineElement
    {
        public readonly string Name;
        public readonly PipelineElementType ElementType;
        public readonly IDictionary<string, object> Properties;

        public PipelineElement(string name, PipelineElementType elementType,
            IDictionary<string, object> properties)
        {
            Name = name;
            ElementType = elementType;
            Properties = properties;
        }
    }
}
