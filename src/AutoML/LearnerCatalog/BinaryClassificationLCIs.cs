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
        private const int DefaultNumIterations = 10;

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildAveragePerceptronParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            Action<AveragedPerceptronTrainer.Arguments> argsFunc = null;
            if (sweepParams == null)
            {
                argsFunc = (args) =>
                {
                    args.NumIterations = DefaultNumIterations;
                };
            }
            else
            {
                argsFunc = LearnerCatalogUtil.CreateArgsFunc<AveragedPerceptronTrainer.Arguments>(sweepParams);
            }
            return mlContext.BinaryClassification.Trainers.AveragedPerceptron(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.AveragedPerceptronBinary;
        }
    }

    public class FastForestBinaryClassificationLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastForestParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<FastForestClassification.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.FastForest(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.FastForestBinary;
        }
    }

    public class FastTreeBinaryClassificationLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastTreeParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<FastTreeBinaryClassificationTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.FastTree(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.FastTreeBinary;
        }
    }

    public class LightGbmBinaryClassificationLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLightGbmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            Action<LightGbmArguments> argsFunc = LearnerCatalogUtil.CreateLightGbmArgsFunc(sweepParams);
            return mlContext.BinaryClassification.Trainers.LightGbm(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.LightGbmBinary;
        }
    }

    public class LinearSvmBinaryClassificationLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLinearSvmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<LinearSvm.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.LinearSupportVectorMachines(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.LinearSvmBinary;
        }
    }

    public class SdcaBinaryClassificationLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSdcaParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<SdcaBinaryTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.SdcaBinary;
        }
    }

    public class LogisticRegressionBinaryClassificationLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLogisticRegressionParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<LogisticRegression.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.LogisticRegression(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.LogisticRegressionBinary;
        }
    }

    public class SgdBinaryClassificationLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSgdParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<StochasticGradientDescentClassificationTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.StochasticGradientDescent(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.StochasticGradientDescentBinary;
        }
    }

    public class SymSgdBinaryClassificationLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSymSgdParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<SymSgdClassificationTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.SymbolicStochasticGradientDescent(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.SymSgdBinary;
        }
    }
}