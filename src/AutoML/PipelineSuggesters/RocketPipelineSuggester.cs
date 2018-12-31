// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.PipelineInference2
{
    internal class RocketPipelineSuggester : PipelineSuggesterBase
    {
        private const int TopKLearners = 1;
        private const int SecondRoundTrialsPerLearner = 5;

        private int _currentStage;
        private int _remainingStageOneTrials;
        private int _remainingStageTwoTrials;
        private int _currentLearnerIndex;
        private readonly Dictionary<string, ISweeper> _hyperSweepers;

        private enum Stage
        {
            First,
            Second,
            Third
        }

        public RocketPipelineSuggester(MLContext mlContext, bool isMaximizingMetric) : base(mlContext, isMaximizingMetric)
        {
            _currentStage = (int)Stage.First;
            _hyperSweepers = new Dictionary<string, ISweeper>();
        }

        public override void UpdateLearners(IEnumerable<SuggestedLearner> availableLearners)
        {
            base.UpdateLearners(availableLearners);
            switch(_currentStage)
            {
                case (int)Stage.First:
                    _remainingStageOneTrials = AvailableLearners.Count();
                    break;
                case (int)Stage.Second:
                    _remainingStageTwoTrials = AvailableLearners.Count() * SecondRoundTrialsPerLearner;
                    break;
            }
        }

        public override IEnumerable<PipelinePattern> GetNextPipelines(IEnumerable<PipelinePattern> history, int numberOfCandidates)
        {
            var pipelnes = new List<PipelinePattern>();
            for (var i = 0; i < numberOfCandidates; i++)
            {
                var pipeline = GetNextPipeline(history);
                pipelnes.Add(pipeline);
            }
            return pipelnes;
        }

        private IEnumerable<SuggestedLearner> GetTopLearners(IEnumerable<PipelinePattern> history)
        {
            history = history.GroupBy(h => h.Learner.LearnerName).Select(g => g.First());
            IEnumerable<PipelinePattern> sortedHistory = history.OrderBy(h => h.Result);
            if(IsMaximizingMetric)
            {
                sortedHistory = sortedHistory.Reverse();
            }
            var topLearners = sortedHistory.Take(TopKLearners).Select(h => h.Learner);
            return topLearners;
        }

        private PipelinePattern GetNextPipeline(IEnumerable<PipelinePattern> history)
        {
            PipelinePattern pipeline;

            if (_currentStage == (int)Stage.First)
            {
                pipeline = GetNextStageOnePipeline();
            }
            else
            {
                var learner = AvailableLearners.ElementAt(_currentLearnerIndex).Clone();
                _currentLearnerIndex = (_currentLearnerIndex + 1) % AvailableLearners.Count();

                SampleHyperparameters(learner, history);

                // make sure we have not seen pipeline before.
                // repeat until passes or runs out of chances.
                const int maxNumberAttempts = 10;
                var count = 0;
                bool valid;
                do
                {
                    pipeline = new PipelinePattern(AvailableTransforms, learner, MLContext);
                    valid = !VisitedPipelines.Contains(pipeline.ToString()) && !HasPipelineFailed(pipeline);
                    count++;
                } while (!valid && count <= maxNumberAttempts);

                // keep only valid pipelines
                if (valid)
                {
                    VisitedPipelines.Add(pipeline.ToString());
                }
            }

            IncrementStageState(history);

            return pipeline;
        }
        
        private void IncrementStageState(IEnumerable<PipelinePattern> history)
        {
            switch (_currentStage)
            {
                case (int)Stage.First:
                    _remainingStageOneTrials--;
                    if (_remainingStageOneTrials == 0)
                    {
                        _currentStage++;

                        // select top k learners, update stage, then get requested
                        // number of candidates, using second stage logic
                        var topLearners = GetTopLearners(history);
                        UpdateLearners(topLearners);
                        foreach(var learner in topLearners)
                        {
                            InitHyperparamSweepers(learner);
                        }

                        _currentLearnerIndex = 0;
                    }
                    break;

                case (int)Stage.Second:
                    _remainingStageTwoTrials--;
                    if(_remainingStageTwoTrials == 0)
                    {
                        _currentStage++;
                    }
                    break;
            }
        }

        private PipelinePattern GetNextStageOnePipeline()
        {
            var learner = AvailableLearners.ElementAt(_currentLearnerIndex);
            var pipeline = new PipelinePattern(AvailableTransforms, learner, MLContext);

            // update current index
            _currentLearnerIndex = (_currentLearnerIndex + 1) % AvailableLearners.Count();

            return pipeline;
        }

        /// <summary>
        /// if first time optimizing hyperparams, create new hyperparameter sweepers
        /// </summary>
        private void InitHyperparamSweepers(SuggestedLearner learner)
        {
            if (!_hyperSweepers.ContainsKey(learner.LearnerName))
            {
                var sps = AutoMlUtils.ConvertToValueGenerators(learner.SweepParams);
                _hyperSweepers[learner.LearnerName] = new KdoSweeper(
                    new KdoSweeper.Arguments
                    {
                        SweptParameters = sps,
                        NumberInitialPopulation = SecondRoundTrialsPerLearner
                    });
            }
        }

        private void SampleHyperparameters(SuggestedLearner learner, IEnumerable<PipelinePattern> history)
        {
            var sweeper = _hyperSweepers[learner.LearnerName];
            IEnumerable<PipelinePattern> historyToUse = new PipelinePattern[0];
            if (_currentStage == (int)Stage.Third)
            {
                historyToUse = history.Where(p => p.Learner.LearnerName == learner.LearnerName && p.Learner.HyperParamSet != null);
            }

            // get new set of hyperparameter values
            var proposedParamSet = sweeper.ProposeSweeps(1, AutoMlUtils.ConvertToRunResults(history, IsMaximizingMetric)).First();

            // associate proposed param set with learner, so that smart hyperparam
            // sweepers (like KDO) can map them back.
            learner.SetHyperparamValues(proposedParamSet);
        }
    }
}