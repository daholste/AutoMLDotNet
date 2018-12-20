using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime;
using System.Linq;
using static Microsoft.ML.Runtime.PipelineInference2.AutoInference;

namespace Microsoft.ML.PipelineInference2
{
    public class AutoMLResult
    {
        public ITransformer BestModel;
        public IEnumerable<PipelinePattern> AllPipelines;
    }

    public static class MLContextExtensions
    {
        public static AutoMLResult Auto(this BinaryClassificationContext context,
            IDataView trainData, IDataView validationData, int maxIterations, IEstimator<ITransformer> preprocessor)
        {
            // hack: init new MLContext
            var mlContext = new MLContext();

            // preprocess train and validation data
            var preprocessorTransform = preprocessor.Fit(trainData);
            trainData = preprocessorTransform.Transform(trainData);
            validationData = preprocessorTransform.Transform(validationData);

            var rocketEngine = new RocketEngine(mlContext, new RocketEngine.Arguments() { });
            var terminator = new IterationTerminator(maxIterations);

            var amls = new AutoMlMlState(mlContext,
                PipelineSweeperSupportedMetrics.GetSupportedMetric(PipelineSweeperSupportedMetrics.Metrics.Accuracy),
                rocketEngine, terminator, MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer,
                   trainData, validationData);
            var pipelineResults = amls.InferPipelines(1, 1, 100);

            // hack: start dummy host & channel
            var host = (mlContext as IHostEnvironment).Register("hi");
            var ch = host.Start("hi");

            var bestPipeline = pipelineResults.First();
            // hack: retrain on best iteration
            var bestPipelineTransformer = bestPipeline.TrainTransformer(trainData);

            // prepend preprocessors to AutoML model before returning
            var bestModel = preprocessorTransform.Append(bestPipelineTransformer);

            return new AutoMLResult()
            {
                BestModel = bestModel,
                AllPipelines = pipelineResults
            };
        }
    }
}
