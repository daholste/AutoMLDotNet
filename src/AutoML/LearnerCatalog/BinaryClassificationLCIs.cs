using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.Trainers.SymSgd;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    using ITrainerEstimator = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor>;

    public class AveragedPerceptronBinaryClassificationLCI : ILearnerCatalogItem
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
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<AveragedPerceptronTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.AveragedPerceptron(advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return LearnerNames.AveragedPerceptron.ToString();
        }
    }

    public class FastForestBinaryClassifierLCI : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> _sweepRanges = LearnerCatalogUtil.TreeArgsSweepableParams;

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<FastForestClassification.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.FastForest(advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "FastForest";
        }
    }

    public class FastTreeBinaryClassifierLCI : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> _sweepRanges = LearnerCatalogUtil.TreeArgsSweepableParams;

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<FastTreeBinaryClassificationTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.FastTree(advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "FastTree";
        }
    }

    public class LightGbmBinaryClassificationLCI : ILearnerCatalogItem
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

                // TreeBoster params
                new SweepableDiscreteParam("RegLambda", new object[] { 0f, 0.5f, 1f }),
                new SweepableDiscreteParam("RegAlpha", new object[] { 0f, 0.5f, 1f })
            };

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            Action<LightGbmArguments> argsFunc = null;
            if (sweepParams != null)
            {
                argsFunc = (args) => {
                    AutoMlUtils.UpdatePropertiesAndFields(args, sweepParams);
                    AutoMlUtils.UpdatePropertiesAndFields(args.Booster, sweepParams);
                };
            }
            return mlContext.BinaryClassification.Trainers.LightGbm(advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "LightGbm";
        }
    }

    public class LinearSvmBinaryClassificationLCI : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> _sweepRanges = new SweepableParam[] {
            new SweepableFloatParam("Lambda", 0.00001f, 0.1f, 10, isLogScale: true),
            new SweepableDiscreteParam("PerformProjection", null, isBool: true),
            new SweepableDiscreteParam("NoBias", null, isBool: true)
        }.Concat(LearnerCatalogUtil.OnlineLinearArgsSweepableParams);

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<LinearSvm.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.LinearSupportVectorMachines(advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "LinearSvm";
        }
    }

    public class SdcaBinaryClassificationLCI : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> _sweepRanges = new SweepableParam[] {
            new SweepableDiscreteParam("L2Const", new object[] { "<Auto>", 1e-7f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f }),
            new SweepableDiscreteParam("L1Threshold", new object[] { "<Auto>", 0f, 0.25f, 0.5f, 0.75f, 1f }),
            new SweepableDiscreteParam("ConvergenceTolerance", new object[] { 0.001f, 0.01f, 0.1f, 0.2f }),
            new SweepableDiscreteParam("MaxIterations", new object[] { "<Auto>", 10, 20, 100 }),
            new SweepableDiscreteParam("Shuffle", null, isBool: true),
            new SweepableDiscreteParam("BiasLearningRate", new object[] { 0.0f, 0.01f, 0.1f, 1f })
        };

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<SdcaBinaryTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "Sdca";
        }
    }

    public class LogisticRegressionBinaryClassificationLCI : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> _sweepRanges = new SweepableParam[] {
            new SweepableFloatParam("L2Weight", 0.0f, 1.0f, numSteps: 4),
            new SweepableFloatParam("L1Weight", 0.0f, 1.0f, numSteps: 4),
            new SweepableDiscreteParam("OptTol", new object[] { 1e-4f, 1e-7f }),
            new SweepableDiscreteParam("MemorySize", new object[] { 5, 20, 50 }),
            new SweepableLongParam("MaxIterations", 1, int.MaxValue),
            new SweepableFloatParam("InitWtsDiameter", 0.0f, 1.0f, numSteps: 5),
            new SweepableDiscreteParam("DenseOptimizer", new object[] { false, true }),
        };

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<LogisticRegression.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.LogisticRegression(advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "LogisticRegression";
        }
    }

    public class SgdBinaryClassificationLCI : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> _sweepRanges = new SweepableParam[] {
            new SweepableDiscreteParam("L2Const", new object[] { 1e-7f, 5e-7f, 1e-6f, 5e-6f, 1e-5f }),
            new SweepableDiscreteParam("ConvergenceTolerance", new object[] { 1e-2f, 1e-3f, 1e-4f, 1e-5f }),
            new SweepableDiscreteParam("MaxIterations", new object[] { 1, 5, 10, 20 }),
            new SweepableDiscreteParam("Shuffle", null, isBool: true),
        };

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<StochasticGradientDescentClassificationTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.StochasticGradientDescent(advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "StochasticGradientDescent";
        }
    }

    public class SymSgdBinaryClassificationLCI : ILearnerCatalogItem
    {
        private static readonly IEnumerable<SweepableParam> _sweepRanges = new SweepableParam[] {
            new SweepableDiscreteParam("NumberOfIterations", new object[] { 1, 5, 10, 20, 30, 40, 50 }),
            new SweepableDiscreteParam("LearningRate", new object[] { "<Auto>", 1e1f, 1e0f, 1e-1f, 1e-2f, 1e-3f }),
            new SweepableDiscreteParam("L2Regularization", new object[] { 0.0f, 1e-5f, 1e-5f, 1e-6f, 1e-7f }),
            new SweepableDiscreteParam("UpdateFrequency", new object[] { "<Auto>", 5, 20 })
        };

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _sweepRanges;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<SymSgdClassificationTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.SymbolicStochasticGradientDescent(advancedSettings: argsFunc);
        }

        public string GetLearnerName()
        {
            return "SymSGD";
        }
    }
}
