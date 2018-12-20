using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.Online;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    public interface ILearnerCatalogItem
    {
        IEnumerable<SweepableParam> GetHyperparamSweepRanges();
        ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams);
        string GetLearnerName();
    }

    public class AveragedPerceptronCatalogItem : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> _sweepRanges =
            LearnerCatalogUtil.AveragedLinearArgsSweepableParams
                .Concat(LearnerCatalogUtil.OnlineLinearArgsSweepableParams);

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            Action<AveragedPerceptronTrainer.Arguments> argsFunc = null;
            if (sweepParams != null)
            {
                argsFunc = (obj) => AutoMlUtils.UpdatePropertiesAndFields(obj, sweepParams);
            }
            return mlContext.BinaryClassification.Trainers.AveragedPerceptron(advancedSettings: argsFunc);
            //return new AveragedPerceptronTrainer(mlContext, advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "AveragedPerceptron";
        }
    }

    public class FastForestCatalogItem : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> _sweepRanges = LearnerCatalogUtil.TreeArgsSweepableParams;

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            Action<FastForestClassification.Arguments> argsFunc = null;
            if (sweepParams != null)
            {
                argsFunc = (obj) => AutoMlUtils.UpdatePropertiesAndFields(obj, sweepParams);
            }
            return new FastForestClassification(mlContext, advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "FastForest";
        }
    }

    public class FastTreeBinaryClassifierCatalogItem : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> _sweepRanges = LearnerCatalogUtil.TreeArgsSweepableParams;

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            Action<FastTreeBinaryClassificationTrainer.Arguments> argsFunc = null;
            if (sweepParams != null)
            {
                argsFunc = (obj) => AutoMlUtils.UpdatePropertiesAndFields(obj, sweepParams);
            }
            return new FastTreeBinaryClassificationTrainer(mlContext, advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "FastTreeBinaryClassifier";
        }
    }

    public class LightGbmBinaryTrainerCatalogItem : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> _sweepRanges = new SweepableParam[]
            {
                new SweepableDiscreteParam("NumBoostRound", new object[] { 10, 20, 50, 100, 150, 200 }),
                new SweepableFloatParam("LearningRate", 0.025f, 0.4f, isLogScale: true),
                new SweepableLongParam("NumLeaves", 2, 128, isLogScale: true, stepSize: 4),
                new SweepableDiscreteParam("MinDataPerLeaf", new object[] { 1, 10, 20, 50 }),
                new SweepableDiscreteParam("UseSoftmax", new object[] { true, false }),
                new SweepableDiscreteParam("UseCat", new object[] { true, false }),
                new SweepableDiscreteParam("UseMissing", new object[] { true, false }),
                new SweepableDiscreteParam("MinDataPerGroup", new object[] { 10, 50, 100, 200 }),
                new SweepableDiscreteParam("MaxCatThreshold", new object[] { 8, 16, 32, 64 }),
                new SweepableDiscreteParam("CatSmooth", new object[] { 1, 10, 20 }),
                new SweepableDiscreteParam("CatL2", new object[] { 0.1, 0.5, 1, 5, 10 }),

            };

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            Action<FastTreeBinaryClassificationTrainer.Arguments> argsFunc = null;
            if (sweepParams != null)
            {
                argsFunc = (obj) => AutoMlUtils.UpdatePropertiesAndFields(obj, sweepParams);
            }
            return new FastTreeBinaryClassificationTrainer(mlContext, advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "LightGbmBinaryTrainer";
        }
    }
}
