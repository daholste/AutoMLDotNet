using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime;
using System.Linq;
using static Microsoft.ML.PipelineInference2.AutoInference;

namespace Microsoft.ML.PipelineInference2
{
    public class AutoMLResult
    {
        public ITransformer BestModel;
        public IEnumerable<PipelinePattern> AllPipelines;
    }

    public static class MLContextExtensions
    {
        public static AutoMLResult AutoFit(this BinaryClassificationContext context,
            IDataView trainData, IDataView validationData, int maxIterations, IEstimator<ITransformer> preprocessor = null)
        {
            return AutoFit(trainData, validationData, maxIterations, preprocessor, MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer,
                OptimizingMetric.Accuracy);
        }

        public static AutoMLResult AutoFit(this MulticlassClassificationContext context,
            IDataView trainData, IDataView validationData, int maxIterations, IEstimator<ITransformer> preprocessor = null)
        {
            return AutoFit(trainData, validationData, maxIterations, preprocessor, MacroUtils.TrainerKinds.SignatureMultiClassClassifierTrainer,
                OptimizingMetric.Accuracy);
        }

        public static AutoMLResult AutoFit(IDataView trainData, IDataView validationData, int maxIterations, IEstimator<ITransformer> preprocessor,
            MacroUtils.TrainerKinds task, OptimizingMetric metric)
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
            var rocketEngine = new RocketPipelineSuggester(mlContext, optimizingMetricfInfo.IsMaximizing);
            var terminator = new IterationBasedTerminator(maxIterations);

            var auotFitter = new AutoFitter(mlContext, optimizingMetricfInfo, terminator, rocketEngine, task,
                   maxIterations, trainData, validationData);
            var pipelineResults = auotFitter.InferPipelines(1, 1, 100);

            var bestPipeline = pipelineResults.First();
            // hack: retrain on best iteration
            var model = bestPipeline.TrainTransformer(trainData);

            if (preprocessorTransform != null)
            {
                // prepend preprocessors to AutoML model before returning
                model = preprocessorTransform.Append(model);
            }

            return new AutoMLResult()
            {
                BestModel = model,
                AllPipelines = pipelineResults
            };
        }
    }
}
