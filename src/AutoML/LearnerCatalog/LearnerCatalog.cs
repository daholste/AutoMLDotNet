using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    public class LearnerCatalog
    {
        public static IEnumerable<ILearnerCatalogItem> GetLearners(MacroUtils.TrainerKinds trainerKind, int maxNumIterations)
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

        private static IEnumerable<ILearnerCatalogItem> GetBinaryLearners(int maxNumIterations)
        {
            var learners = new List<ILearnerCatalogItem>()
            {
                new AveragedPerceptronBinaryClassificationLCI(),
                new SdcaBinaryClassificationLCI(),
                new LightGbmBinaryClassificationLCI(),
                new SymSgdBinaryClassificationLCI()
            };

            if(maxNumIterations < 20)
            {
                return learners;
            }

            learners.AddRange(new ILearnerCatalogItem[] {
                new LinearSvmBinaryClassificationLCI(),
                new FastTreeBinaryClassificationLCI()
            });

            if(maxNumIterations < 100)
            {
                return learners;
            }

            learners.AddRange(new ILearnerCatalogItem[] {
                new LogisticRegressionBinaryClassificationLCI(),
                new FastForestBinaryClassificationLCI(),
                new SgdBinaryClassificationLCI()
            });

            return learners;
        }

        private static IEnumerable<ILearnerCatalogItem> GetMultiLearners(int maxNumIterations)
        {
            var learners = new List<ILearnerCatalogItem>()
            {
                new AveragedPerceptronOvaLCI(),
                new SdcaMulticlassClassificationLCI(),
                new LightGbmMulticlassClassificationLCI(),
                new SymSgdOvaLCI()
            };

            if (maxNumIterations < 20)
            {
                return learners;
            }

            learners.AddRange(new ILearnerCatalogItem[] {
                new FastTreeOvaLCI(),
                new LinearSvmOvaLCI(),
                new LogisticRegressionOvaLCI()
            });

            if (maxNumIterations < 100)
            {
                return learners;
            }

            learners.AddRange(new ILearnerCatalogItem[] {
                new SgdBinaryClassificationLCI(),
                new FastForestBinaryClassificationLCI(),
                new LogisticRegressionBinaryClassificationLCI(),
            });

            return learners;
        }

        private static IEnumerable<ILearnerCatalogItem> GetRegressionLearners()
        {
            return new ILearnerCatalogItem[]
            {
                new FastForestRegressionLCI(),
                new FastTreeRegressionLCI(),
                new FastTreeTweedieRegressionLCI(),
                new LightGbmRegressionLCI(),
                new OnlineGradientDescentRegressionLCI(),
                new OrdinaryLeastSquaresRegressionLCI(),
                new PoissonRegressionLCI(),
                new SdcaRegressionLCI()
            };
        }
    }
}
