using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Training;

namespace Microsoft.ML.Auto
{
    internal class SuggestedTrainer
    {
        public IEnumerable<SweepableParam> SweepParams { get; }
        public string TrainerName { get; }
        public ParameterSet HyperParamSet { get; set; }

        private readonly MLContext _mlContext;
        private readonly ITrainerExtension _trainerExtension;

        internal SuggestedTrainer(MLContext mlContext, ITrainerExtension trainerExtension,
            ParameterSet hyperParamSet = null)
        {
            _mlContext = mlContext;
            _trainerExtension = trainerExtension;
            SweepParams = _trainerExtension.GetHyperparamSweepRanges();
            TrainerName = _trainerExtension.GetTrainerName().ToString();
            SetHyperparamValues(hyperParamSet);
        }

        public void SetHyperparamValues(ParameterSet hyperParamSet)
        {
            HyperParamSet = hyperParamSet;
            PropagateParamSetValues();
        }

        public SuggestedTrainer Clone()
        {
            return new SuggestedTrainer(_mlContext, _trainerExtension, HyperParamSet?.Clone());
        }

        public ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> BuildTrainer(MLContext env)
        {
            IEnumerable<SweepableParam> sweepParams = null;
            if (HyperParamSet != null)
            {
                sweepParams = SweepParams;
            }
            return _trainerExtension.CreateInstance(_mlContext, sweepParams);
        }

        public override string ToString()
        {
            var paramsStr = string.Empty;
            if (SweepParams != null)
            {
                paramsStr = string.Join(", ", SweepParams.Where(p => p != null && p.RawValue != null).Select(p => $"{p.Name}:{p.ProcessedValue()}"));
            }
            return $"{TrainerName}{{{paramsStr}}}";
        }

        public Public.PipelineElement ToObjectModel()
        {
            var hyperParams = SweepParams.Where(p => p != null && p.RawValue != null);
            var elementProperties = new Dictionary<string, object>();
            foreach (var hyperParam in hyperParams)
            {
                elementProperties[hyperParam.Name] = hyperParam.ProcessedValue();
            }
            return new Public.PipelineElement(TrainerName, Public.PipelineElementType.Trainer, elementProperties);
        }

        /// <summary>
        /// make sure sweep params and param set are consistent
        /// </summary>
        private void PropagateParamSetValues()
        {
            if (HyperParamSet == null)
            {
                return;
            }

            var spMap = SweepParams.ToDictionary(sp => sp.Name);

            foreach (var hp in HyperParamSet)
            {
                if (spMap.ContainsKey(hp.Name))
                {
                    var sp = spMap[hp.Name];
                    sp.SetUsingValueText(hp.ValueText);
                }
            }
        }
    }
}
