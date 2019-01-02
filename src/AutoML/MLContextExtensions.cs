using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime;
using System.Linq;
using static Microsoft.ML.Auto.AutoFitter;

namespace Microsoft.ML.Auto
{
    public class AutoMLResult
    {
        public ITransformer BestModel;
        public IEnumerable<Auto.ObjectModel.Pipeline> AllPipelines;
    }

    public static class MLContextExtensions
    {
        public static AutoMLResult AutoFit(this BinaryClassificationContext context,
            IDataView trainData, IDataView validationData, int maxIterations, IEstimator<ITransformer> preprocessor = null)
        {
            return AutoFit(trainData, validationData, maxIterations, preprocessor, TaskKind.BinaryClassification,
                OptimizingMetric.Accuracy);
        }

        public static AutoMLResult AutoFit(this MulticlassClassificationContext context,
            IDataView trainData, IDataView validationData, int maxIterations, IEstimator<ITransformer> preprocessor = null)
        {
            return AutoFit(trainData, validationData, maxIterations, preprocessor, TaskKind.MulticlassClassification,
                OptimizingMetric.Accuracy);
        }

        public static AutoMLResult AutoFit(IDataView trainData, IDataView validationData, int maxIterations, IEstimator<ITransformer> preprocessor,
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

            var optimizingMetricfInfo = new OptimizingMetricInfo(OptimizingMetric.Accuracy);
            var terminator = new IterationBasedTerminator(maxIterations);

            var auotFitter = new AutoFitter(mlContext, optimizingMetricfInfo, terminator, task,
                   maxIterations, trainData, validationData);
            var (pipelineResults, models, bestModel) = auotFitter.InferPipelines(1, 1, 100);

            var bestPipeline = pipelineResults.First();

            if (preprocessorTransform != null)
            {
                // prepend preprocessors to AutoML model before returning
                bestModel = preprocessorTransform.Append(bestModel);
            }

            return new AutoMLResult()
            {
                BestModel = bestModel,
                AllPipelines = pipelineResults
            };
        }
    }
}
