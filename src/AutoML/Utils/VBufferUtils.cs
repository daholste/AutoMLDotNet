using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    public class VBufferUtils
    {
        public static bool HasNaNs(in VBuffer<Single> buffer)
        {
            var values = buffer.GetValues();
            for (int i = 0; i < values.Length; i++)
            {
                if (Single.IsNaN(values[i]))
                    return true;
            }
            return false;
        }
    }
}
