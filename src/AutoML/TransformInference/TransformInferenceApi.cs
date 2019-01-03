using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Auto
{
    internal static class TransformInferenceApi
    {
        public static IEnumerable<SuggestedTransform> InferTransforms(MLContext mlContext, IDataView data)
        {
            var args = new TransformInference.Arguments
            {
                EstimatedSampleFraction = 1.0,
                ExcludeFeaturesConcatTransforms = false
            };
            return TransformInference.InferTransforms(mlContext, data, args);
        }
    }
}
