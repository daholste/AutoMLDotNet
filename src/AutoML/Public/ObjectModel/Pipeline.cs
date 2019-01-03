using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Auto.Public
{
    public class Pipeline
    {
        public readonly PipelineElement[] Elements;

        public Pipeline(PipelineElement[] elements)
        {
            Elements = elements;
        }
    }
}
