using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.HalLearners;
using Microsoft.ML.Trainers.Online;

namespace Microsoft.ML.PipelineInference2
{
    using ITrainerEstimator = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor>;

    public class FastForestRegressionLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.FastForest;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<FastForestRegression.Arguments>(sweepParams);
            return mlContext.Regression.Trainers.FastForest(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.FastForestRegression;
        }
    }

    public class FastTreeRegressionLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.FastTree;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<FastTreeRegressionTrainer.Arguments>(sweepParams);
            return mlContext.Regression.Trainers.FastTree(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.FastTreeRegression;
        }
    }

    public class FastTreeTweedieRegressionLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.FastTreeTweedie;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<FastTreeTweedieTrainer.Arguments>(sweepParams);
            return mlContext.Regression.Trainers.FastTreeTweedie(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.FastTreeTweedieRegression;
        }
    }

    public class LightGbmRegressionLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.LightGbm;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateLightGbmArgsFunc(sweepParams);
            return mlContext.Regression.Trainers.LightGbm(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.LightGbmRegression;
        }
    }

    public class OnlineGradientDescentRegressionLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.OnlineGradientDescent;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<AveragedLinearArguments>(sweepParams);
            return mlContext.Regression.Trainers.OnlineGradientDescent(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.OnlineGradientDescentRegression;
        }
    }

    public class OrdinaryLeastSquaresRegressionLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.OrdinaryLeastSquares;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<OlsLinearRegressionTrainer.Arguments>(sweepParams);
            return mlContext.Regression.Trainers.OrdinaryLeastSquares(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.OrdinaryLeastSquaresRegression;
        }
    }

    public class PoissonRegressionLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.PoissonRegression;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<PoissonRegression.Arguments>(sweepParams);
            return mlContext.Regression.Trainers.PoissonRegression(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.PoissonRegression;
        }
    }

    public class SdcaRegressionLCI : ILearnerCatalogItem
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.Sdca;
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = LearnerCatalogUtil.CreateArgsFunc<SdcaRegressionTrainer.Arguments>(sweepParams);
            return mlContext.Regression.Trainers.StochasticDualCoordinateAscent(advancedSettings: argsFunc);
        }

        public LearnerName GetLearnerName()
        {
            return LearnerName.SdcaRegression;
        }
    }
}