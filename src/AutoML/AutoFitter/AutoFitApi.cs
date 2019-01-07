using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Auto
{
    internal static class AutoFitApi
    {
        public static (PipelineRunResult[] allPipelines, PipelineRunResult bestPipeline) AutoFit(IDataView trainData, 
            IDataView validationData, string label, int maxIterations, IEstimator<ITransformer> preprocessor, 
            TaskKind task, OptimizingMetric metric)
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
            var auotFitter = new AutoFitter(mlContext, optimizingMetricfInfo, terminator, task,
                   maxIterations, label, trainData, validationData);
            var allPipelines = auotFitter.InferPipelines(1);

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
    }
}
