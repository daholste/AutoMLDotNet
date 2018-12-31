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
    public class SuggestedLearner
    {
        public IEnumerable<SweepableParam> SweepParams { get; }
        public string LearnerName { get; }
        public ParameterSet HyperParamSet { get; set; }

        private readonly MLContext _mlContext;
        private readonly ITrainerExtension _learnerCatalogItem;

        internal SuggestedLearner(MLContext mlContext, ITrainerExtension learnerCatalogItem,
            ParameterSet hyperParamSet = null)
        {
            _mlContext = mlContext;
            _learnerCatalogItem = learnerCatalogItem;
            SweepParams = _learnerCatalogItem.GetHyperparamSweepRanges();
            LearnerName = _learnerCatalogItem.GetTrainerName().ToString();
            SetHyperparamValues(hyperParamSet);
        }

        public void SetHyperparamValues(ParameterSet hyperParamSet)
        {
            HyperParamSet = hyperParamSet;
            PropagateParamSetValues();
        }

        public SuggestedLearner Clone()
        {
            return new SuggestedLearner(_mlContext, _learnerCatalogItem, HyperParamSet?.Clone());
        }

        public ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> BuildTrainer(MLContext env)
        {
            IEnumerable<SweepableParam> sweepParams = null;
            if(HyperParamSet != null)
            {
                sweepParams = SweepParams;
            }
            return _learnerCatalogItem.CreateInstance(_mlContext, sweepParams);
        }

        public override string ToString()
        {
            var paramsStr = string.Empty;
            if(SweepParams != null)
            {
                paramsStr = string.Join(", ", SweepParams.Where(p => p != null && p.RawValue != null).Select(p => $"{p.Name}:{p.ProcessedValue()}"));
            }
            return $"{LearnerName}{{{paramsStr}}}";
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
        public readonly struct SuggestedRecipe
        {
            public readonly TransformInference.SuggestedTransform[] Transforms;
            
            internal readonly IEnumerable<SuggestedLearner> Learners;
        }

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
        /// Given a predictor type returns a set of all permissible learners (with their sweeper params, if defined).
        /// </summary>
        /// <returns>Array of viable learners.</returns>
        public static IEnumerable<SuggestedLearner> AllowedLearners(MLContext mlContext, MacroUtils.TrainerKinds task,
            int maxNumIterations)
        {
            var learnerCatalogItems = TrainerExtensionCatalog.GetTrainers(task, maxNumIterations);

            var learners = new List<SuggestedLearner>();
            foreach (var learnerCatalogItem in learnerCatalogItems)
            {
                var learner = new SuggestedLearner(mlContext, learnerCatalogItem);
                learners.Add(learner);
            }
            return learners.ToArray();
        }
    }
}
