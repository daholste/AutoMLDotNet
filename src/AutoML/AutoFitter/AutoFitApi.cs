using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class AutoFitApi
    {
        public static (PipelineRunResult[] allPipelines, PipelineRunResult bestPipeline) Fit(IDataView trainData, 
            IDataView validationData, string label, InferredColumn[] inferredColumns, AutoFitSettings settings, 
            TaskKind task, OptimizingMetric metric, IDebugLogger debugLogger)
        {
            // hack: init new MLContext
            var mlContext = new MLContext();

            // infer pipelines
            var optimizingMetricfInfo = new OptimizingMetricInfo(metric);
            var autoFitter = new AutoFitter(mlContext, optimizingMetricfInfo, settings, task,
                   label, ToInternalColumnPurposes(inferredColumns), 
                   trainData, validationData, debugLogger);
            var allPipelines = autoFitter.Fit(1);

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
