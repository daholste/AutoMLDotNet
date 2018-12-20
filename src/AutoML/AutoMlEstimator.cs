using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;
using static Microsoft.ML.Core.Data.SchemaShape;
using static Microsoft.ML.Core.Data.SchemaShape.Column;
using Microsoft.ML.Runtime;
using static Microsoft.ML.RegressionContext;
using System.Linq;
using Microsoft.ML.PipelineInference2;
using static Microsoft.ML.BinaryClassificationContext;
using static Microsoft.ML.Runtime.PipelineInference2.AutoInference;

namespace Microsoft.ML.PipelineInference2
{
    public static class AutoMlExtension
    {
        public static AutoMlBinaryClassificationEstimator Auto(this BinaryClassificationTrainers trainers,
            int maxIterations = 10, IDataView validationData = null)
        {
            // hack: init new MLContext
            var mlContext = new MLContext();

            return new AutoMlBinaryClassificationEstimator(mlContext, maxIterations, validationData);
        }
    }

    public class AutoMlBinaryClassificationEstimator : IEstimator<ITransformer>
    {
        private readonly MLContext _env;
        private readonly int _maxIterations;

        public IDataView ValidationData { get; set; }

        public AutoMlBinaryClassificationEstimator(MLContext env, int maxIterations = 10, IDataView validationData = null)
        {
            _env = env;
            _maxIterations = maxIterations;
            ValidationData = validationData;
        }

        public ITransformer Fit(IDataView trainData)
        {
            var rocketEngine = new RocketEngine(_env, new RocketEngine.Arguments() { });
            var terminator = new IterationTerminator(_maxIterations);

            var amls = new AutoMlMlState(_env,
                PipelineSweeperSupportedMetrics.GetSupportedMetric(PipelineSweeperSupportedMetrics.Metrics.Accuracy),
                rocketEngine, terminator, MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer,
                   trainData, ValidationData);
            var bestPipelines = amls.InferPipelines(1, 3, 100);
            var bestPipeline = bestPipelines.First();

            var transformer = bestPipeline.TrainTransformer(trainData);
            return transformer;
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            // get list of existing columns
            var cols = inputSchema.Columns.ToList();

            // add output columns
            cols.AddRange(new[] {
                new SchemaShape.Column("Probability", VectorKind.Scalar, NumberType.R4, false),
                new SchemaShape.Column("PredictedLabel", VectorKind.Scalar, BoolType.Instance, false)
            });

            return new SchemaShape(cols);
        }
    }
}
