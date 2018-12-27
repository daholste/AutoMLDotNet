using Microsoft.ML.Runtime.LightGBM;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    public enum LearnerName
    {
        AveragedPerceptronBinary,
        AveragedPerceptronOva,
        FastForestBinary,
        FastForestOva,
        FastForestRegression,
        FastTreeBinary,
        FastTreeOva,
        FastTreeRegression,
        FastTreeTweedieRegression,
        LightGbmBinary,
        LightGbmMulti,
        LightGbmRegression,
        LinearSvmBinary,
        LinearSvmOva,
        LogisticRegressionBinary,
        LogisticRegressionOva,
        LogisticRegressionMulti,
        OnlineGradientDescentRegression,
        OrdinaryLeastSquaresRegression,
        PoissonRegression,
        SdcaBinary,
        SdcaMulti,
        SdcaRegression,
        StochasticGradientDescentBinary,
        StochasticGradientDescentOva,
        SymSgdBinary,
        SymSgdOva
    }

    public static class LearnerCatalogUtil
    {
        public static Action<T> CreateArgsFunc<T>(IEnumerable<SweepableParam> sweepParams)
        {
            Action<T> argsFunc = null;
            if (sweepParams != null)
            {
                argsFunc = (args) => AutoMlUtils.UpdateFields(args, sweepParams);
            }
            return argsFunc;
        }

        private static string[] _treeBoosterParamNames = new[] { "RegLambda", "RegAlpha" };

        public static Action<LightGbmArguments> CreateLightGbmArgsFunc(IEnumerable<SweepableParam> sweepParams)
        {
            var treeBoosterParams = sweepParams.Where(p => _treeBoosterParamNames.Contains(p.Name));
            var parentArgParams = sweepParams.Except(treeBoosterParams);

            Action<LightGbmArguments> argsFunc = null;
            if (sweepParams != null)
            {
                argsFunc = (args) =>
                {
                    AutoMlUtils.UpdateFields(args, sweepParams);
                    AutoMlUtils.UpdateFields(args.Booster, sweepParams);
                };
            }
            return argsFunc;
        }
    }
}
