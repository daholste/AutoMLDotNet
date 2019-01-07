using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Auto
{
    internal static class TransformInferenceApi
    {
        public static IEnumerable<SuggestedTransform> InferTransforms(MLContext context, IDataView data, string label)
        {
            // infer column purposes
            var columnIndices = Enumerable.Range(0, data.Schema.Count);
            var purposes = PurposeInference.InferPurposes(context, data, columnIndices, label);

            return TransformInference.InferTransforms(context, data, purposes);
        }
    }
}
