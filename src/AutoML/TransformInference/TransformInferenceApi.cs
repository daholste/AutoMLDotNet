using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Auto
{
    internal static class TransformInferenceApi
    {
        public static IEnumerable<SuggestedTransform> InferTransforms(MLContext context, IDataView data, string label, 
            PurposeInference.Column[] purposeOverrides = null)
        {
            return AutoMlUtils.ExecuteApiFuncSafe(InferenceType.Transform, () =>
                InferTransformsSafe(context, data, label, purposeOverrides));
        }

        private static IEnumerable<SuggestedTransform> InferTransformsSafe(MLContext context, IDataView data, string label,
            PurposeInference.Column[] purposeOverrides = null)
        {
            // infer column purposes
            var purposes = PurposeInference.InferPurposes(context, data, label, purposeOverrides);

            return TransformInference.InferTransforms(context, data, purposes);
        }
    }
}
