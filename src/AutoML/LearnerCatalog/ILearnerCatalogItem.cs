using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Training;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    public interface ILearnerCatalogItem
    {
        IEnumerable<SweepableParam> GetHyperparamSweepRanges();
        ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams);
        string GetLearnerName();
    }
}
