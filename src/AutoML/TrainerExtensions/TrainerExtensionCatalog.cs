// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    internal class TrainerExtensionCatalog
    {
        public static IEnumerable<ITrainerExtension> GetTrainers(MacroUtils.TrainerKinds trainerKind, int maxNumIterations)
        {
            if(trainerKind == MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer)
            {
                return GetBinaryLearners(maxNumIterations);
            }
            else if (trainerKind == MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer)
            {
                return GetMultiLearners(maxNumIterations);
            }
            else if (trainerKind == MacroUtils.TrainerKinds.SignatureRegressorTrainer)
            {
                return GetRegressionLearners();
            }
            else
            {
                // todo: fix this up
                throw new Exception("unsupported task");
            }
        }

        private static IEnumerable<ITrainerExtension> GetBinaryLearners(int maxNumIterations)
        {
            var learners = new List<ITrainerExtension>()
            {
                new AveragedPerceptronBinaryExtension(),
                new SdcaBinaryExtension(),
                new LightGbmBinaryExtension(),
                new SymSgdBinaryExtension()
            };

            if(maxNumIterations < 20)
            {
                return learners;
            }

            learners.AddRange(new ITrainerExtension[] {
                //new LinearSvmBinaryExtension(),
                new FastTreeBinaryExtension()
            });

            if(maxNumIterations < 100)
            {
                return learners;
            }

            learners.AddRange(new ITrainerExtension[] {
                new LogisticRegressionBinaryExtension(),
                new FastForestBinaryExtension(),
                new SgdBinaryExtension()
            });

            return learners;
        }

        private static IEnumerable<ITrainerExtension> GetMultiLearners(int maxNumIterations)
        {
            var learners = new List<ITrainerExtension>()
            {
                new AveragedPerceptronOvaExtension(),
                new SdcaMultiExtension(),
                new LightGbmMultiExtension(),
                new SymSgdOvaExtension()
            };

            if (maxNumIterations < 20)
            {
                return learners;
            }

            learners.AddRange(new ITrainerExtension[] {
                new FastTreeOvaExtension(),
                new LinearSvmOvaExtension(),
                new LogisticRegressionOvaExtension()
            });

            if (maxNumIterations < 100)
            {
                return learners;
            }

            learners.AddRange(new ITrainerExtension[] {
                new SgdOvaExtension(),
                new FastForestOvaExtension(),
                new LogisticRegressionMultiExtension(),
            });

            return learners;
        }

        private static IEnumerable<ITrainerExtension> GetRegressionLearners()
        {
            return new ITrainerExtension[]
            {
                new FastForestRegressionExtension(),
                new FastTreeRegressionExtension(),
                new FastTreeTweedieRegressionExtension(),
                new LightGbmRegressionExtension(),
                new OnlineGradientDescentRegressionExtension(),
                new OrdinaryLeastSquaresRegressionExtension(),
                new PoissonRegressionExtension(),
                new SdcaRegressionExtension()
            };
        }
    }
}
