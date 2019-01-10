// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Auto
{
    internal static class PipelineSuggester
    {
        private const int TopKTrainers = 3;

        public static InferredPipeline GetNextPipeline(IEnumerable<PipelineRunResult> history,
            IEnumerable<SuggestedTransform> transforms,
            IEnumerable<SuggestedTrainer> availableTrainers,
            bool isMaximizingMetric = true)
        {
            // if we haven't run all pipelines once
            if(history.Count() < availableTrainers.Count())
            {
                return GetNextFirstStagePipeline(history, availableTrainers, transforms);
            }

            // get next trainer
            var topTrainers = GetTopTrainers(history, availableTrainers, isMaximizingMetric);
            var nextTrainerIndex = (history.Count() - availableTrainers.Count()) % topTrainers.Count();
            var trainer = topTrainers.ElementAt(nextTrainerIndex).Clone();

            // make sure we have not seen pipeline before.
            // repeat until passes or runs out of chances.
            var visitedPipelines = new HashSet<InferredPipeline>(history.Select(h => h.Pipeline));
            const int maxNumberAttempts = 10;
            var count = 0;
            do
            {
                SampleHyperparameters(trainer, history, isMaximizingMetric);
                var pipeline = new InferredPipeline(transforms, trainer);
                if(!visitedPipelines.Contains(pipeline))
                {
                    return pipeline;
                }
            } while (++count <= maxNumberAttempts);

            return null;
        }
        
        /// <summary>
        /// Get top trainers from first stage
        /// </summary>
        private static IEnumerable<SuggestedTrainer> GetTopTrainers(IEnumerable<PipelineRunResult> history, 
            IEnumerable<SuggestedTrainer> availableTrainers,
            bool isMaximizingMetric)
        {
            // narrow history to first stage runs
            history = history.Take(availableTrainers.Count());

            history = history.GroupBy(r => r.Pipeline.Trainer.TrainerName).Select(g => g.First());
            IEnumerable<PipelineRunResult> sortedHistory = history.OrderBy(r => r.Score);
            if(isMaximizingMetric)
            {
                sortedHistory = sortedHistory.Reverse();
            }
            var topTrainers = sortedHistory.Take(TopKTrainers).Select(r => r.Pipeline.Trainer);
            return topTrainers;
        }

        private static InferredPipeline GetNextFirstStagePipeline(IEnumerable<PipelineRunResult> history,
            IEnumerable<SuggestedTrainer> availableTrainers,
            IEnumerable<SuggestedTransform> transforms)
        {
            var trainer = availableTrainers.ElementAt(history.Count());
            return new InferredPipeline(transforms, trainer);
        }

        private static IValueGenerator[] ConvertToValueGenerators(IEnumerable<SweepableParam> hps)
        {
            var results = new IValueGenerator[hps.Count()];

            for (int i = 0; i < hps.Count(); i++)
            {
                switch (hps.ElementAt(i))
                {
                    case SweepableDiscreteParam dp:
                        var dpArgs = new DiscreteParamArguments()
                        {
                            Name = dp.Name,
                            Values = dp.Options.Select(o => o.ToString()).ToArray()
                        };
                        results[i] = new DiscreteValueGenerator(dpArgs);
                        break;

                    case SweepableFloatParam fp:
                        var fpArgs = new FloatParamArguments()
                        {
                            Name = fp.Name,
                            Min = fp.Min,
                            Max = fp.Max,
                            LogBase = fp.IsLogScale,
                        };
                        if (fp.NumSteps.HasValue)
                        {
                            fpArgs.NumSteps = fp.NumSteps.Value;
                        }
                        if (fp.StepSize.HasValue)
                        {
                            fpArgs.StepSize = fp.StepSize.Value;
                        }
                        results[i] = new FloatValueGenerator(fpArgs);
                        break;

                    case SweepableLongParam lp:
                        var lpArgs = new LongParamArguments()
                        {
                            Name = lp.Name,
                            Min = lp.Min,
                            Max = lp.Max,
                            LogBase = lp.IsLogScale
                        };
                        if (lp.NumSteps.HasValue)
                        {
                            lpArgs.NumSteps = lp.NumSteps.Value;
                        }
                        if (lp.StepSize.HasValue)
                        {
                            lpArgs.StepSize = lp.StepSize.Value;
                        }
                        results[i] = new LongValueGenerator(lpArgs);
                        break;
                }
            }
            return results;
        }

        private static void SampleHyperparameters(SuggestedTrainer trainer, IEnumerable<PipelineRunResult> history, bool isMaximizingMetric)
        {
            var sps = ConvertToValueGenerators(trainer.SweepParams);
            var sweeper = new SmacSweeper(
                new SmacSweeper.Arguments
                {
                    SweptParameters = sps
                });

            IEnumerable<PipelineRunResult> historyToUse = history
                .Where(r => r.RunSucceded && r.Pipeline.Trainer.TrainerName == trainer.TrainerName && r.Pipeline.Trainer.HyperParamSet != null);

            // get new set of hyperparameter values
            var proposedParamSet = sweeper.ProposeSweeps(1, historyToUse.Select(h => h.ToRunResult(isMaximizingMetric))).First();

            // associate proposed param set with trainer, so that smart hyperparam
            // sweepers (like KDO) can map them back.
            trainer.SetHyperparamValues(proposedParamSet);
        }
    }
}