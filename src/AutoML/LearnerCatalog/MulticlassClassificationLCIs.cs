using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Online;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    using ITrainerEstimatorProducingFloat = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>;
    using ITrainerEstimator = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor>;
    
    public class AveragedPerceptronOvaLCI : ILearnerCatalogItem
    {
        private static readonly ILearnerCatalogItem _binaryLearnerCatalogItem = new AveragedPerceptronBinaryClassificationLCI();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.AveragePerceptron;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.AveragedPerceptronOva;
        }
    }

    public class FastForestOvaLCI : ILearnerCatalogItem
    {
        private static readonly ILearnerCatalogItem _binaryLearnerCatalogItem = new FastForestBinaryClassificationLCI();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.FastForest;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.FastForestOva;
        }
    }

    public class LightGbmMulticlassClassificationLCI : ILearnerCatalogItem
    {
        private static readonly ILearnerCatalogItem _binaryLearnerCatalogItem = new LightGbmBinaryClassificationLCI();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.LightGbm;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            Action<LightGbmArguments> argsFunc = LearnerCatalogUtil.CreateLightGbmArgsFunc(sweepParams);
            return mlContext.MulticlassClassification.Trainers.LightGbm(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.LightGbmMulti;
        }
    }

    public class LinearSvmOvaLCI : ILearnerCatalogItem
    {
        private static readonly ILearnerCatalogItem _binaryLearnerCatalogItem = new LinearSvmBinaryClassificationLCI();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.LinearSvm;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.LinearSvmOva;
        }
    }

    public class SdcaMulticlassClassificationLCI : ILearnerCatalogItem
    {
        private static readonly ILearnerCatalogItem _binaryLearnerCatalogItem = new SdcaBinaryClassificationLCI();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.Sdca;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<SdcaMultiClassTrainer.Arguments>(sweepParams);
            return mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.SdcaMulti;
        }
    }


    public class LogisticRegressionOvaLCI : ILearnerCatalogItem
    {
        private static readonly ILearnerCatalogItem _binaryLearnerCatalogItem = new LogisticRegressionBinaryClassificationLCI();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.LogisticRegression;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.LogisticRegressionOva;
        }
    }

    public class SgdOvaLCI : ILearnerCatalogItem
    {
        private static readonly ILearnerCatalogItem _binaryLearnerCatalogItem = new SgdBinaryClassificationLCI();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.Sgd;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.StochasticGradientDescentOva;
        }
    }

    public class SymSgdOvaLCI : ILearnerCatalogItem
    {
        private static readonly ILearnerCatalogItem _binaryLearnerCatalogItem = new SymSgdBinaryClassificationLCI();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _binaryLearnerCatalogItem.GetHyperparamSweepRanges();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.SymSgdOva;
        }
    }

    public class FastTreeOvaLCI : ILearnerCatalogItem
    {
        private static readonly ILearnerCatalogItem _binaryLearnerCatalogItem = new FastTreeBinaryClassificationLCI();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.FastTree;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.FastTreeOva;
        }
    }

    public class LogisticRegressionMulticlassClassificationLCI : ILearnerCatalogItem
    {
        private static readonly ILearnerCatalogItem _binaryLearnerCatalogItem = new LogisticRegressionBinaryClassificationLCI();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.LogisticRegression;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<MulticlassLogisticRegression.Arguments>(sweepParams);
            return mlContext.MulticlassClassification.Trainers.LogisticRegression(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.LogisticRegressionMulti;
        }
    }
}