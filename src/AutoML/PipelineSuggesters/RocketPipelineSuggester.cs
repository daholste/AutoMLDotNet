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
        private const int TopKTrainers = 2;
        private const int SecondStageTrialsPerTrainer = 2;

        private readonly Dictionary<string, ISweeper> _hyperSweepers;
        private readonly ISet<Pipeline> _visitedPipelines;

        private int _currentTrainerIndex;
        private int _currentStage;
        private int _remainingTrialsInCurrStage;

        private enum Stage
        {
            First,
            Second,
            Third
        }

        public RocketPipelineSuggester(MLContext mlContext, bool isMaximizingMetric, 
            IEnumerable<SuggestedTrainer> availableTrainers, IEnumerable<SuggestedTransform> availableTransforms) 
            : base(mlContext, isMaximizingMetric, availableTrainers, availableTransforms)
        {
            _currentStage = (int)Stage.First;
            _hyperSweepers = new Dictionary<string, ISweeper>();
            _visitedPipelines = new HashSet<Pipeline>();
            _remainingTrialsInCurrStage = availableTrainers.Count();
        }

        public override IEnumerable<Pipeline> GetNextPipelines(IEnumerable<PipelineRunResult> history, int requestedBatchSize)
        {
            MoveToNextStageIfNeeded(history);

            var batchSize = GetBatchSize(requestedBatchSize);

            var pipelnes = new List<Pipeline>();
            for (var i = 0; i < batchSize; i++)
            {
                var pipeline = GetNextPipeline(history);
                pipelnes.Add(pipeline);
                _remainingTrialsInCurrStage--;
            }
            return pipelnes;
        }

        private IEnumerable<SuggestedTrainer> GetTopTrainers(IEnumerable<PipelineRunResult> history)
        {
            history = history.GroupBy(r => r.Pipeline.Trainer.TrainerName).Select(g => g.First());
            IEnumerable<PipelineRunResult> sortedHistory = history.OrderBy(r => r.Result);
            if(IsMaximizingMetric)
            {
                sortedHistory = sortedHistory.Reverse();
            }
            var topTrainers = sortedHistory.Take(TopKTrainers).Select(r => r.Pipeline.Trainer);
            return topTrainers;
        }

        private Pipeline GetNextPipeline(IEnumerable<PipelineRunResult> history)
        {
            Pipeline pipeline;

            if (_currentStage == (int)Stage.First)
            {
                pipeline = GetNextFirstStagePipeline();
            }
            else
            {
                var trainer = AvailableTrainers.ElementAt(_currentTrainerIndex).Clone();
                _currentTrainerIndex = (_currentTrainerIndex + 1) % AvailableTrainers.Count();

                SampleHyperparameters(trainer, history);

                // make sure we have not seen pipeline before.
                // repeat until passes or runs out of chances.
                const int maxNumberAttempts = 10;
                var count = 0;
                bool valid;
                do
                {
                    pipeline = new Pipeline(AvailableTransforms, trainer, MLContext);
                    valid = !_visitedPipelines.Contains(pipeline) && !HasPipelineFailed(pipeline);
                    count++;
                } while (!valid && count <= maxNumberAttempts);

                // keep only valid pipelines
                if (valid)
                {
                    _visitedPipelines.Add(pipeline);
                }
            }

            return pipeline;
        }

        private void MoveToNextStageIfNeeded(IEnumerable<PipelineRunResult> history)
        {
            switch (_currentStage)
            {
                case (int)Stage.First:
                    if (_remainingTrialsInCurrStage != 0)
                    {
                        break;
                    }

                    _currentStage++;

                    // select and use only top trainers from here on out
                    var topTrainers = GetTopTrainers(history);
                    AvailableTrainers = topTrainers;

                    foreach (var trainer in topTrainers)
                    {
                        InitHyperparamSweepers(trainer);
                    }

                    // reset current trainer index
                    _currentTrainerIndex = 0;

                    // set # of trials for second stage
                    _remainingTrialsInCurrStage = TopKTrainers * SecondStageTrialsPerTrainer;

                    break;

                case (int)Stage.Second:
                    if (_remainingTrialsInCurrStage != 0)
                    {
                        break;
                    }

                    _currentStage++;

                    break;
            }
        }

        private Pipeline GetNextFirstStagePipeline()
        {
            var trainer = AvailableTrainers.ElementAt(_currentTrainerIndex);
            var pipeline = new Pipeline(AvailableTransforms, trainer, MLContext);

            // update current index
            _currentTrainerIndex++;

            return pipeline;
        }

        /// <summary>
        /// if first time optimizing hyperparams, create new hyperparameter sweepers
        /// </summary>
        private void InitHyperparamSweepers(SuggestedTrainer trainer)
        {
            var sps = AutoMlUtils.ConvertToValueGenerators(trainer.SweepParams);
            _hyperSweepers[trainer.TrainerName] = new KdoSweeper(
                new KdoSweeper.Arguments
                {
                    SweptParameters = sps,
                    NumberInitialPopulation = SecondStageTrialsPerTrainer
                });
        }

        private void SampleHyperparameters(SuggestedTrainer trainer, IEnumerable<PipelineRunResult> history)
        {
            var sweeper = _hyperSweepers[trainer.TrainerName];
            IEnumerable<PipelineRunResult> historyToUse = new PipelineRunResult[0];
            if (_currentStage == (int)Stage.Third)
            {
                historyToUse = history.Where(r => r.Pipeline.Trainer.TrainerName == trainer.TrainerName && r.Pipeline.Trainer.HyperParamSet != null);
            }

            // get new set of hyperparameter values
            var proposedParamSet = sweeper.ProposeSweeps(1, AutoMlUtils.ConvertToRunResults(historyToUse, IsMaximizingMetric)).First();

            // associate proposed param set with trainer, so that smart hyperparam
            // sweepers (like KDO) can map them back.
            trainer.SetHyperparamValues(proposedParamSet);
        }

        private int GetBatchSize(int requestedBatchSize)
        {
            if (_currentStage == (int)Stage.Third)
            {
                return requestedBatchSize;
            }
            return Math.Min(_remainingTrialsInCurrStage, requestedBatchSize);
        }
    }
}