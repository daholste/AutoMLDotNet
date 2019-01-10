using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class TransformInferenceApi
    {
        public static IEnumerable<SuggestedTransform> InferTransforms(MLContext context, IDataView data, string label, 
            PurposeInference.Column[] purposeOverrides = null)
        {
            // infer column purposes
            var purposes = PurposeInference.InferPurposes(context, data, label, purposeOverrides);

            return TransformInference.InferTransforms(context, data, purposes);
        }
    }
}
