using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    public class LearnerCatalog
    {
        public static LearnerCatalog Instance = new LearnerCatalog();

        private static readonly IDictionary<MacroUtils.TrainerKinds, IEnumerable<ILearnerCatalogItem>> _tasksToLearners =
            new Dictionary<MacroUtils.TrainerKinds, IEnumerable<ILearnerCatalogItem>>()
            {
                { MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer,
                    new ILearnerCatalogItem[] {
                        new AveragedPerceptronCatalogItem(),
                        new FastForestCatalogItem(),
                        new FastTreeBinaryClassifierCatalogItem(),
                        new LightGbmBinaryTrainerCatalogItem()
                    } },
            };

        public IEnumerable<ILearnerCatalogItem> GetLearners(MacroUtils.TrainerKinds trainerKind)
        {
            return _tasksToLearners[trainerKind];
        }
    }
}
