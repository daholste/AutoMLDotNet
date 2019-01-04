// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.LightGBM;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Auto
{
    public enum TrainerName
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

    internal static class TrainerExtensionUtil
    {
        public static Action<T> CreateArgsFunc<T>(IEnumerable<SweepableParam> sweepParams)
        {
            Action<T> argsFunc = null;
            if (sweepParams != null)
            {
                argsFunc = (args) =>
                {
                    AutoFitterUtil.UpdateFields(args, sweepParams);
                };
            }
            return argsFunc;
        }

        private static string[] _treeBoosterParamNames = new[] { "RegLambda", "RegAlpha" };

        public static Action<LightGbmArguments> CreateLightGbmArgsFunc(IEnumerable<SweepableParam> sweepParams)
        {
            Action<LightGbmArguments> argsFunc = null;
            if (sweepParams != null)
            {
                argsFunc = (args) =>
                {
                    var treeBoosterParams = sweepParams.Where(p => _treeBoosterParamNames.Contains(p.Name));
                    var parentArgParams = sweepParams.Except(treeBoosterParams);
                    AutoFitterUtil.UpdateFields(args, parentArgParams);
                    AutoFitterUtil.UpdateFields(args.Booster, treeBoosterParams);
                };
            }
            return argsFunc;
        }
    }
}
