using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Auto.ObjectModel
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
