// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Training;

namespace Microsoft.ML.PipelineInference2
{
    public class SuggestedTrainer
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
            if(HyperParamSet != null)
            {
                sweepParams = SweepParams;
            }
            return _trainerExtension.CreateInstance(_mlContext, sweepParams);
        }

        public override string ToString()
        {
            var paramsStr = string.Empty;
            if(SweepParams != null)
            {
                paramsStr = string.Join(", ", SweepParams.Where(p => p != null && p.RawValue != null).Select(p => $"{p.Name}:{p.ProcessedValue()}"));
            }
            return $"{TrainerName}{{{paramsStr}}}";
        }

        public Auto.ObjectModel.PipelineElement ToObjectModel()
        {
            var hyperParams = SweepParams.Where(p => p != null && p.RawValue != null);
            var elementProperties = new Dictionary<string, object>();
            foreach(var hyperParam in hyperParams)
            {
                elementProperties[hyperParam.Name] = hyperParam.ProcessedValue();
            }
            return new Auto.ObjectModel.PipelineElement(TrainerName, Auto.ObjectModel.PipelineElementType.Trainer, elementProperties);
        }

        /// <summary>
        /// make sure sweep params and param set are consistent
        /// </summary>
        private void PropagateParamSetValues()
        {
            if(HyperParamSet == null)
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

    public static class RecipeInference
    {
        public static TextLoader.Arguments MyAutoMlInferTextLoaderArguments(MLContext env,
            string dataFile, string labelColName)
        {
            var sample = TextFileSample.CreateFromFullFile(dataFile);
            var splitResult = TextFileContents.TrySplitColumns(sample, TextFileContents.DefaultSeparators);
            var columnPurposes = InferenceUtils.InferColumnPurposes(env, sample, splitResult, out var hasHeader, labelColName);
            return new TextLoader.Arguments
            {
                Column = ColumnGroupingInference.GenerateLoaderColumns(columnPurposes),
                HasHeader = true,
                Separator = splitResult.Separator,
                AllowSparse = splitResult.AllowSparse,
                AllowQuoting = splitResult.AllowQuote
            };
        }

        /// <summary>
        /// Given a predictor type & target max num of iterations, return a set of all permissible trainers (with their sweeper params, if defined).
        /// </summary>
        /// <returns>Array of viable learners.</returns>
        public static IEnumerable<SuggestedTrainer> AllowedTrainers(MLContext mlContext, MacroUtils.TrainerKinds task,
            int maxNumIterations)
        {
            var trainerExtensions = TrainerExtensionCatalog.GetTrainers(task, maxNumIterations);

            var trainers = new List<SuggestedTrainer>();
            foreach (var trainerExtension in trainerExtensions)
            {
                var learner = new SuggestedTrainer(mlContext, trainerExtension);
                trainers.Add(learner);
            }
            return trainers.ToArray();
        }
    }
}
