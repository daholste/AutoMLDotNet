// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.PipelineInference2
{
    /// <summary>
    /// Interface that defines what an autoML engine looks like, so that it can plug into the
    /// autoML body defined in this file. Will allow us to change autoML techniques or use third-
    /// party services as needed, just by swapping out the pipeline optimizer used.
    /// </summary>
    public interface IPipelineOptimizer
    {
        PipelinePattern[] GetNextCandidates(IEnumerable<PipelinePattern> history, int numberOfCandidates);

        void SetSpace(TransformInference.SuggestedTransform[] availableTransforms,
            RecipeInference.SuggestedRecipe.SuggestedLearner[] availableLearners,
            IDataView originalData, IDataView fullyTransformedData,
            AutoInference.DependencyMap dependencyMapping,
            bool isMaximizingMetric);

        void UpdateLearners(RecipeInference.SuggestedRecipe.SuggestedLearner[] availableLearners);
    }

    public abstract class PipelineOptimizerBase : IPipelineOptimizer
    {
        protected TransformInference.SuggestedTransform[] AvailableTransforms;
        protected RecipeInference.SuggestedRecipe.SuggestedLearner[] AvailableLearners;
        protected IDataView OriginalData;
        protected IDataView FullyTransformedData;
        protected AutoInference.DependencyMap DependencyMapping;
        protected readonly MLContext Env;
        protected readonly Dictionary<long, bool> TransformsMaskValidity;
        protected readonly HashSet<string> VisitedPipelines;
        protected readonly SweeperProbabilityUtils ProbUtils;
        protected bool IsMaximizingMetric;

        protected PipelineOptimizerBase(MLContext env)
        {
            Env = env;
            TransformsMaskValidity = new Dictionary<long, bool>();
            VisitedPipelines = new HashSet<string>();
            ProbUtils = new SweeperProbabilityUtils();
        }

        public abstract PipelinePattern[] GetNextCandidates(IEnumerable<PipelinePattern> history, int numberOfCandidates);

        public virtual void SetSpace(TransformInference.SuggestedTransform[] availableTransforms,
            RecipeInference.SuggestedRecipe.SuggestedLearner[] availableLearners,
            IDataView originalData, IDataView fullyTransformedData, AutoInference.DependencyMap dependencyMapping,
            bool isMaximizingMetric)
        {
            AvailableLearners = availableLearners;
            AvailableTransforms = availableTransforms;
            OriginalData = originalData;
            FullyTransformedData = fullyTransformedData;
            DependencyMapping = dependencyMapping;
            IsMaximizingMetric = isMaximizingMetric;
        }

        public virtual void UpdateLearners(RecipeInference.SuggestedRecipe.SuggestedLearner[] availableLearners)
        {
            AvailableLearners = availableLearners;
        }

        protected string GetHashKey(long transformsBitMask, RecipeInference.SuggestedRecipe.SuggestedLearner learner)
        {
            var learnerName = learner.ToString();
            //Host.Check(!string.IsNullOrEmpty(learnerName));
            return $"{learnerName}+{transformsBitMask}";
        }

        protected double[] LearnerHistoryToWeights(PipelinePattern[] history, bool isMaximizingMetric)
        {
            int numLearners = AvailableLearners.Length;
            double[] weights = new double[numLearners];
            int[] counts = new int[numLearners];
            Dictionary<string, int> labelToIndex = new Dictionary<string, int>();
            double maxWeight = history.Length > 0 ? history.Max(w => w.PerformanceSummary.MetricValue) : 0d;

            // Map categorical values to their index
            for (int j = 0; j < numLearners; j++)
                labelToIndex[AvailableLearners[j].LearnerName] = j;

            // Add mass according to performance
            foreach (var pipeline in history)
            {
                if (AvailableLearners.All(l => l.LearnerName != pipeline.Learner.LearnerName))
                    continue;
                weights[labelToIndex[pipeline.Learner.LearnerName]] +=
                    AutoMlUtils.ProcessWeight(pipeline.PerformanceSummary.MetricValue,
                        maxWeight, isMaximizingMetric);
                counts[labelToIndex[pipeline.Learner.LearnerName]]++;
            }

            // Take average mass for each learner
            for (int i = 0; i < weights.Length; i++)
                weights[i] /= counts[i] > 0 ? counts[i] : 1;

            // If any learner has not been seen, default its average to 1.0
            // to encourage exploration of untried algorithms.
            for (int i = 0; i < weights.Length; i++)
                weights[i] += counts[i] == 0 ? 1 : 0;

            // Normalize weights to sum to one and return
            return SweeperProbabilityUtils.Normalize(weights);
        }

        protected void SampleHyperparameters(RecipeInference.SuggestedRecipe.SuggestedLearner learner, ISweeper sweeper,
            bool isMaximizingMetric, PipelinePattern[] history)
        {
            // Make sure there are hyperparameters to sweep over.
            var hyperParams = learner.PipelineNode.SweepParams;
            if (!hyperParams.Any())
                return;

            // Get new set of hyperparameter values.
            var proposedParamSet = sweeper.ProposeSweeps(1, AutoMlUtils.ConvertToRunResults(history, isMaximizingMetric)).First();
            //Env.Assert(proposedParamSet != null && proposedParamSet.All(ps => hyperParams.Any(hp => hp.Name == ps.Name)));

            // Associate proposed param set with learner, so that smart hyperparam
            // sweepers (like KDO) can map them back.
            learner.PipelineNode.HyperSweeperParamSet = proposedParamSet;

            var generatorSet = hyperParams.Select(AutoMlUtils.ToIValueGenerator).ToArray();
            var values = SweeperProbabilityUtils.ParameterSetAsFloatArray(generatorSet, proposedParamSet, false);

            // Update hyperparameters.
            for (int i = 0; i < hyperParams.Count(); i++)
            {
                if (hyperParams.ElementAt(i) is SweepableDiscreteParam dp)
                    hyperParams.ElementAt(i).RawValue = (int)values[i];
                else
                    hyperParams.ElementAt(i).RawValue = values[i];
            }
        }
    }

    public class FalseSweeper : ISweeper
    {
        public ParameterSet[] ProposeSweeps(int numCandidates, IEnumerable<IRunResult> history) => new ParameterSet[0];
    }
}
