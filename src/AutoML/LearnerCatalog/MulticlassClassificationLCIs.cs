using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Trainers.Online;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    using ITrainerEstimatorProducingFloat = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>;
    using ITrainerEstimator = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor>;
    
    public class AveragedPerceptronMultiClassificationLCI : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> _sweepRanges =
        LearnerCatalogUtil.AveragedLinearArgsSweepableParams
            .Concat(LearnerCatalogUtil.OnlineLinearArgsSweepableParams);

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = new AveragedPerceptronBinaryClassificationLCI().CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public string GetLearnerName()
        {
            return LearnerNames.AveragedPerceptron.ToString();
        }
    }
}
