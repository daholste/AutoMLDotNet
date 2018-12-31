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

        public override void UpdateTrainers(IEnumerable<SuggestedTrainer> availableLearners)
        {
            base.UpdateTrainers(availableLearners);
            switch(_currentStage)
            {
                case (int)Stage.First:
                    _remainingStageOneTrials = AvailableTrainers.Count();
                    break;
                case (int)Stage.Second:
                    _remainingStageTwoTrials = AvailableTrainers.Count() * SecondRoundTrialsPerLearner;
                    break;
            }
        }

        public override IEnumerable<Pipeline> GetNextPipelines(IEnumerable<Pipeline> history, int numberOfCandidates)
        {
            var pipelnes = new List<Pipeline>();
            for (var i = 0; i < numberOfCandidates; i++)
            {
                var pipeline = GetNextPipeline(history);
                pipelnes.Add(pipeline);
            }
            return pipelnes;
        }

        private IEnumerable<SuggestedTrainer> GetTopLearners(IEnumerable<Pipeline> history)
        {
            history = history.GroupBy(h => h.Trainer.TrainerName).Select(g => g.First());
            IEnumerable<Pipeline> sortedHistory = history.OrderBy(h => h.Result);
            if(IsMaximizingMetric)
            {
                sortedHistory = sortedHistory.Reverse();
            }
            var topLearners = sortedHistory.Take(TopKLearners).Select(h => h.Trainer);
            return topLearners;
        }

        private Pipeline GetNextPipeline(IEnumerable<Pipeline> history)
        {
            Pipeline pipeline;

            if (_currentStage == (int)Stage.First)
            {
                pipeline = GetNextStageOnePipeline();
            }
            else
            {
                var learner = AvailableTrainers.ElementAt(_currentLearnerIndex).Clone();
                _currentLearnerIndex = (_currentLearnerIndex + 1) % AvailableTrainers.Count();

                SampleHyperparameters(learner, history);

                // make sure we have not seen pipeline before.
                // repeat until passes or runs out of chances.
                const int maxNumberAttempts = 10;
                var count = 0;
                bool valid;
                do
                {
                    pipeline = new Pipeline(AvailableTransforms, learner, MLContext);
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
        
        private void IncrementStageState(IEnumerable<Pipeline> history)
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
                        UpdateTrainers(topLearners);
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

        private Pipeline GetNextStageOnePipeline()
        {
            var learner = AvailableTrainers.ElementAt(_currentLearnerIndex);
            var pipeline = new Pipeline(AvailableTransforms, learner, MLContext);

            // update current index
            _currentLearnerIndex = (_currentLearnerIndex + 1) % AvailableTrainers.Count();

            return pipeline;
        }

        /// <summary>
        /// if first time optimizing hyperparams, create new hyperparameter sweepers
        /// </summary>
        private void InitHyperparamSweepers(SuggestedTrainer learner)
        {
            if (!_hyperSweepers.ContainsKey(learner.TrainerName))
            {
                var sps = AutoMlUtils.ConvertToValueGenerators(learner.SweepParams);
                _hyperSweepers[learner.TrainerName] = new KdoSweeper(
                    new KdoSweeper.Arguments
                    {
                        SweptParameters = sps,
                        NumberInitialPopulation = SecondRoundTrialsPerLearner
                    });
            }
        }

        private void SampleHyperparameters(SuggestedTrainer learner, IEnumerable<Pipeline> history)
        {
            var sweeper = _hyperSweepers[learner.TrainerName];
            IEnumerable<Pipeline> historyToUse = new Pipeline[0];
            if (_currentStage == (int)Stage.Third)
            {
                historyToUse = history.Where(p => p.Trainer.TrainerName == learner.TrainerName && p.Trainer.HyperParamSet != null);
            }

            // get new set of hyperparameter values
            var proposedParamSet = sweeper.ProposeSweeps(1, AutoMlUtils.ConvertToRunResults(history, IsMaximizingMetric)).First();

            // associate proposed param set with learner, so that smart hyperparam
            // sweepers (like KDO) can map them back.
            learner.SetHyperparamValues(proposedParamSet);
        }
    }
}