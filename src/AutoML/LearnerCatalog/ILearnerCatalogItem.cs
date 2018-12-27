using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Training;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    using ITrainerEstimator = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor>;

    public interface ILearnerCatalogItem
    {
        IEnumerable<SweepableParam> GetHyperparamSweepRanges();
        ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams);
        LearnerName GetLearnerName();
    }
}
