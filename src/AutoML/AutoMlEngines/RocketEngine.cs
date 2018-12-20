// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.PipelineInference2;
using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Runtime.Sweeper.Algorithms;

namespace Microsoft.ML.PipelineInference2
{
    public class RocketEngine : PipelineOptimizerBase
    {
        private int _currentStage;
        private int _remainingSecondStageTrials;
        private int _remainingThirdStageTrials;
        private readonly int _topK;
        private readonly bool _randomInit;
        private readonly int _numInitPipelines;
        private readonly Dictionary<string, IPipelineOptimizer> _secondaryEngines;
        private readonly Dictionary<string, ISweeper> _hyperSweepers;
        private enum Stages
        {
            First,
            Second,
            Third
        }

        public sealed class Arguments
        {
            public const int TopKLearners = 3;
            public const int SecondRoundTrialsPerLearner = 5;
            public const bool RandomInitialization = false;

            public int NumInitializationPipelines { get; set; }

            public IPipelineOptimizer CreateComponent(MLContext env) {
                NumInitializationPipelines = new KdoSweeper.Arguments().NumberInitialPopulation;
                return new RocketEngine(env, this);
            }
        }

        public RocketEngine(MLContext env, Arguments args)
            : base(env)
        {
            _currentStage = (int)Stages.First;
            _topK = Arguments.TopKLearners;
            _remainingSecondStageTrials = _topK * Arguments.SecondRoundTrialsPerLearner;
            _remainingThirdStageTrials = 5 * _topK;
            _randomInit = Arguments.RandomInitialization;
            _numInitPipelines = args.NumInitializationPipelines;
            _hyperSweepers = new Dictionary<string, ISweeper>();
            _secondaryEngines = new Dictionary<string, IPipelineOptimizer>
            {
                [nameof(DefaultsEngine)] = new DefaultsEngine(env, new DefaultsEngine.Arguments())
            };
        }

        public override void UpdateLearners(RecipeInference.SuggestedRecipe.SuggestedLearner[] availableLearners)
        {
            foreach (var engine in _secondaryEngines.Values)
                engine.UpdateLearners(availableLearners);
            base.UpdateLearners(availableLearners);
        }

        public override void SetSpace(TransformInference.SuggestedTransform[] availableTransforms,
            RecipeInference.SuggestedRecipe.SuggestedLearner[] availableLearners,
            IDataView originalData, IDataView fullyTransformedData, AutoInference.DependencyMap dependencyMapping,
            bool isMaximizingMetric)
        {
            foreach (var engine in _secondaryEngines.Values)
                engine.SetSpace(availableTransforms, availableLearners,
                    originalData, fullyTransformedData, dependencyMapping,
                    isMaximizingMetric);

            base.SetSpace(availableTransforms, availableLearners, originalData, fullyTransformedData,
                dependencyMapping, isMaximizingMetric);
        }

        private void SampleHyperparameters(RecipeInference.SuggestedRecipe.SuggestedLearner learner, PipelinePattern[] history)
        {
            // If first time optimizing hyperparams, create new hyperparameter sweeper.
            if (!_hyperSweepers.ContainsKey(learner.LearnerName))
            {
                var sps = AutoMlUtils.ConvertToValueGenerators(learner.PipelineNode.SweepParams);
                if (sps.Length > 0)
                {
                    _hyperSweepers[learner.LearnerName] = new KdoSweeper(
                        new KdoSweeper.Arguments
                        {
                            SweptParameters = sps,
                            NumberInitialPopulation = Math.Max(_remainingThirdStageTrials, 2)
                        });
                }
                else
                    _hyperSweepers[learner.LearnerName] = new FalseSweeper();
            }
            var sweeper = _hyperSweepers[learner.LearnerName];
            var historyToUse = history.Where(p => p.Learner.LearnerName == learner.LearnerName).ToArray();
            if (_currentStage == (int)Stages.Third)
            {
                _remainingThirdStageTrials--;
                historyToUse = new PipelinePattern[0];
                if (_remainingThirdStageTrials < 1)
                    _currentStage++;
            }
            SampleHyperparameters(learner, sweeper, IsMaximizingMetric, historyToUse);
        }

        private TransformInference.SuggestedTransform[] SampleTransforms(RecipeInference.SuggestedRecipe.SuggestedLearner learner,
            PipelinePattern[] history, out long transformsBitMask, bool uniformRandomSampling = false)
        {
            // For now, return all transforms.
            var sampledTransforms = AvailableTransforms.ToList();
            transformsBitMask = AutoMlUtils.TransformsToBitmask(sampledTransforms.ToArray());

            //sampledTransforms = sampledTransforms.Take(sampledTransforms.Count() - 3).ToList();

            // Add final features concat transform.
            sampledTransforms.AddRange(AutoMlUtils.GetFinalFeatureConcat(Env, FullyTransformedData,
                DependencyMapping, sampledTransforms.ToArray(), AvailableTransforms));

            return sampledTransforms.ToArray();
        }

        private RecipeInference.SuggestedRecipe.SuggestedLearner[] GetTopLearners(IEnumerable<PipelinePattern> history)
        {
            history = history.GroupBy(h => h.Learner.LearnerName).Select(g => g.First());
            IEnumerable<PipelinePattern> sortedHistory = history.OrderBy(h => h.PerformanceSummary.MetricValue);
            if(IsMaximizingMetric)
            {
                sortedHistory = sortedHistory.Reverse();
            }
            var topLearners = sortedHistory.Take(_topK).Select(h => h.Learner).ToArray();
            return topLearners;
        }

        public override PipelinePattern[] GetNextCandidates(IEnumerable<PipelinePattern> history, int numCandidates)
        {
            var prevCandidates = history.ToArray();

            switch (_currentStage)
            {
                case (int)Stages.First:
                    // First stage: hGo through all learners once with default hyperparams and all transforms.
                    // If random initilization is used, generate number of requested initialization trials for
                    // this stage.
                    int numStageOneTrials = _randomInit ? _numInitPipelines : AvailableLearners.Length;
                    var remainingNum = Math.Min(numStageOneTrials - prevCandidates.Length, numCandidates);
                    if (remainingNum < 1)
                    {
                        // Select top k learners, update stage, then get requested
                        // number of candidates, using second stage logic.
                        UpdateLearners(GetTopLearners(prevCandidates));
                        _currentStage += 2;
                        return GetNextCandidates(prevCandidates, numCandidates);
                    }
                    else
                        return GetInitialPipelines(prevCandidates, remainingNum);
                case (int)Stages.Second:
                    // Second stage: Using top k learners, try random transform configurations.
                    var candidates = new List<PipelinePattern>();
                    var numSecondStageCandidates = Math.Min(numCandidates, _remainingSecondStageTrials);
                    var numThirdStageCandidates = Math.Max(numCandidates - numSecondStageCandidates, 0);
                    _remainingSecondStageTrials -= numSecondStageCandidates;

                    // Get second stage candidates.
                    if (numSecondStageCandidates > 0)
                        candidates.AddRange(NextCandidates(prevCandidates, numSecondStageCandidates, true, true));

                    // Update stage when no more second stage trials to sample.
                    if (_remainingSecondStageTrials < 1)
                        _currentStage++;

                    // If the number of requested candidates is smaller than remaining second stage candidates,
                    // draw candidates from remaining pool.
                    if (numThirdStageCandidates > 0)
                        candidates.AddRange(NextCandidates(prevCandidates, numThirdStageCandidates));

                    return candidates.ToArray();
                default:
                    // Sample transforms according to weights and use hyperparameter optimization method.
                    // Third stage samples hyperparameters uniform randomly in KDO, fourth and above do not.
                    return NextCandidates(prevCandidates, numCandidates);
            }
        }

        private PipelinePattern[] GetInitialPipelines(IEnumerable<PipelinePattern> history, int numCandidates)
        {
            var engine = _secondaryEngines[nameof(DefaultsEngine)];
            return engine.GetNextCandidates(history, numCandidates);
        }

        private PipelinePattern[] NextCandidates(PipelinePattern[] history, int numCandidates,
            bool defaultHyperParams = false, bool uniformRandomTransforms = false)
        {
            const int maxNumberAttempts = 10;
            double[] learnerWeights = LearnerHistoryToWeights(history, IsMaximizingMetric);
            var candidates = new List<PipelinePattern>();
            var sampledLearners = new RecipeInference.SuggestedRecipe.SuggestedLearner[numCandidates];

            if (_currentStage == (int)Stages.Second || _currentStage == (int)Stages.Third)
            {
                // Select remaining learners in round-robin fashion.
                for (int i = 0; i < numCandidates; i++)
                    sampledLearners[i] = AvailableLearners[i % AvailableLearners.Length].Clone();
            }
            else
            {
                // Select learners, based on weights.
                var indices = ProbUtils.SampleCategoricalDistribution(numCandidates, learnerWeights);
                foreach (var item in indices.Select((idx, i) => new { idx, i }))
                    sampledLearners[item.i] = AvailableLearners[item.idx].Clone();
            }

            var totalSkipped = 0;

            // Select hyperparameters and transforms based on learner and history.
            foreach (var learner in sampledLearners)
            {
                PipelinePattern pipeline;
                int count = 0;
                bool valid;
                string hashKey;

                if (!defaultHyperParams)
                    SampleHyperparameters(learner, history);

                do
                {   // Make sure transforms set is valid and have not seen pipeline before.
                    // Repeat until passes or runs out of chances.
                    pipeline = new PipelinePattern(
                        SampleTransforms(learner, history, out var transformsBitMask, uniformRandomTransforms),
                        learner, "", Env);
                    hashKey = GetHashKey(transformsBitMask, learner);
                    valid = !VisitedPipelines.Contains(hashKey) &&
                        !MyGlobals.FailedPipelineHashes.Contains(learner.PipelineNode.ToString());
                    count++;

                    if(count >= maxNumberAttempts)
                    {
                        Console.WriteLine($"{++totalSkipped} got hereee");
                    }
                } while (!valid && count <= maxNumberAttempts);

                // If maxed out chances and at second stage, move onto next stage.
                if (count >= maxNumberAttempts && _currentStage == (int)Stages.Second)
                    _currentStage++;

                // Keep only valid pipelines.
                if (valid)
                {
                    VisitedPipelines.Add(hashKey);
                    candidates.Add(pipeline);
                }
            }
            return candidates.ToArray();
        }
    }
}
