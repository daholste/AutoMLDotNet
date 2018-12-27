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
                return GetBinaryClassificationLearners(maxNumIterations);
            }
            else if (trainerKind == MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer)
            {
                return GetMulticlassClassificationLearners(maxNumIterations);
            }
            else if (trainerKind == MacroUtils.TrainerKinds.SignatureRegressorTrainer)
            {
                return null;
            }
            else
            {
                // todo: fix this up
                throw new Exception("unsupported task");
            }
        }

        private static IEnumerable<ILearnerCatalogItem> GetBinaryClassificationLearners(int maxNumIterations)
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

        private static IEnumerable<ILearnerCatalogItem> GetMulticlassClassificationLearners(int maxNumIterations)
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
    }
}
