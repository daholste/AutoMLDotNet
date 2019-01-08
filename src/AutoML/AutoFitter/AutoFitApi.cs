using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Auto
{
    internal static class AutoFitApi
    {
        public static (PipelineRunResult[] allPipelines, PipelineRunResult bestPipeline) AutoFit(IDataView trainData, 
            IDataView validationData, string label, InferredColumn[] inferredColumns, int maxIterations, 
            IEstimator<ITransformer> preprocessor, TaskKind task, OptimizingMetric metric, IDebugLogger debugLogger)
        {
            // hack: init new MLContext
            var mlContext = new MLContext();

            ITransformer preprocessorTransform = null;
            if (preprocessor != null)
            {
                // preprocess train and validation data
                preprocessorTransform = preprocessor.Fit(trainData);
                trainData = preprocessorTransform.Transform(trainData);
                validationData = preprocessorTransform.Transform(validationData);
            }

            // infer pipelines
            var optimizingMetricfInfo = new OptimizingMetricInfo(metric);
            var terminator = new IterationBasedTerminator(maxIterations);
            var autoFitter = new AutoFitter(mlContext, optimizingMetricfInfo, terminator, task,
                   maxIterations, label, ToInternalColumnPurposes(inferredColumns), 
                   trainData, validationData, debugLogger);
            var allPipelines = autoFitter.InferPipelines(1);

            // apply preprocessor to returned models
            if (preprocessorTransform != null)
            {
                for (var i = 0; i < allPipelines.Length; i++)
                {
                    allPipelines[i].Model = preprocessorTransform.Append(allPipelines[i].Model);
                }
            }

            var bestScore = allPipelines.Max(p => p.Score);
            var bestPipeline = allPipelines.First(p => p.Score == bestScore);

            return (allPipelines, bestPipeline);
        }

        private static PurposeInference.Column[] ToInternalColumnPurposes(InferredColumn[] inferredColumns)
        {
            if (inferredColumns == null)
            {
                return null;
            }

            var result = new List<PurposeInference.Column>();
            foreach(var inferredColumn in inferredColumns)
            {
                result.AddRange(inferredColumn.ToInternalColumnPurposes());
            }
            return result.ToArray();
        }
    }
}
