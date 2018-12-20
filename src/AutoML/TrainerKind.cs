using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    public static class MacroUtils
    {
        /// <summary>
        /// Lists the types of trainer signatures. Used by entry points and autoML system
        /// to know what types of evaluators to use for the train test / pipeline sweeper.
        /// </summary>
        public enum TrainerKinds
        {
            SignatureBinaryClassifierTrainer,
            SignatureMultiClassClassifierTrainer,
            SignatureRankerTrainer,
            SignatureRegressorTrainer,
            SignatureMultiOutputRegressorTrainer,
            SignatureAnomalyDetectorTrainer,
            SignatureClusteringTrainer,
        }
    }
}
