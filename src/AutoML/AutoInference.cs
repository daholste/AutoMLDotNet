// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.IO;
using System.Text;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Core.Data;

namespace Microsoft.ML.PipelineInference2
{
    /// <summary>
    /// Class for generating potential recipes/pipelines, testing them, and zeroing in on the best ones.
    /// For now, only works with maximizing metrics (AUC, Accuracy, etc.).
    /// </summary>
    public class AutoInference
    {
        public sealed class ReversedComparer<T> : IComparer<T>
        {
            public int Compare(T x, T y)
            {
                return Comparer<T>.Default.Compare(y, x);
            }
        }

        /// <summary>
        /// Class that holds state for an autoML search-in-progress. Should be able to resume search, given this object.
        /// </summary>
        internal sealed class AutoFitter
        {
            private readonly SortedList<double, Pipeline> _sortedSampledElements;
            private readonly List<Pipeline> _history;
            private readonly MLContext _mlContext;
            private IDataView _trainData;
            private IDataView _validationData;
            private IDataView _transformedData;
            private ITerminator _terminator;
            private int _maxNumIterations;
            public IPipelineSuggester PipelineSuggester { get; set; }
            public OptimizingMetricInfo OptimizingMetricInfo { get; }
            public MacroUtils.TrainerKinds TrainerKind { get; }

            public AutoFitter(MLContext mlContext, OptimizingMetricInfo metricInfo, ITerminator terminator, IPipelineSuggester pipelineSuggester,
                MacroUtils.TrainerKinds trainerKind, int maxNumIterations,
                IDataView trainData, IDataView validationData)
            {
                OptimizingMetricInfo = metricInfo;
                _sortedSampledElements = OptimizingMetricInfo.IsMaximizing ? new SortedList<double, Pipeline>(new ReversedComparer<double>()) :
                        new SortedList<double, Pipeline>();
                _history = new List<Pipeline>();
                _mlContext = mlContext;
                _trainData = trainData;
                _validationData = validationData;
                _terminator = terminator;
                _maxNumIterations = maxNumIterations;
                PipelineSuggester = pipelineSuggester;
                TrainerKind = trainerKind;
            }

            private void MainLearningLoop(int batchSize)
            {
                var overallExecutionTime = Stopwatch.StartNew();
                var stopwatch = new Stopwatch();
                var probabilityUtils = new SweeperProbabilityUtils();

                while (!_terminator.ShouldTerminate(_history))
                {
                    try
                    {
                        // get next set of candidates
                        var currentBatchSize = batchSize;
                        if (_terminator is IterationBasedTerminator itr)
                        {
                            currentBatchSize = Math.Min(itr.RemainingIterations(_history), batchSize);
                        }
                        var candidates = PipelineSuggester.GetNextPipelines(_sortedSampledElements.Values, currentBatchSize);

                        // break if no candidates returned, means no valid pipeline available
                        if (!candidates.Any())
                        {
                            break;
                        }

                        // evaluate them on subset of data
                        foreach (var candidate in candidates)
                        {
                            try
                            {
                                ProcessPipeline(probabilityUtils, stopwatch, candidate);
                            }
                            catch (Exception e)
                            {
                                File.AppendAllText($"{MyGlobals.OutputDir}/crash_dump1.txt", $"{candidate.Trainer} Crashed {e}\r\n");
                                PipelineSuggester.MarkPipelineAsFailed(candidate);
                                stopwatch.Stop();
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        File.AppendAllText($"{MyGlobals.OutputDir}/crash_dump2.txt", $"{e}\r\n");
                    }
                }
            }

            private void ProcessPipeline(SweeperProbabilityUtils utils, Stopwatch stopwatch, Pipeline candidate)
            {
                // run pipeline, and time how long it takes
                stopwatch.Restart();
                candidate.RunTrainTestExperiment(_trainData, _validationData, TrainerKind, _mlContext, out var testMetricVal);
                stopwatch.Stop();

                // handle key collisions on sorted list
                while (_sortedSampledElements.ContainsKey(testMetricVal))
                {
                    testMetricVal += 1e-10;
                }

                // save performance score
                candidate.Result = testMetricVal;
                _sortedSampledElements.Add(testMetricVal, candidate);
                _history.Add(candidate);

                var transformsSb = new StringBuilder();
                foreach (var transform in candidate.Transforms)
                {
                    transformsSb.Append("xf=");
                    transformsSb.Append(transform);
                    transformsSb.Append(" ");
                }
                var commandLineStr = $"{transformsSb.ToString()} tr={candidate.Trainer}";
                File.AppendAllText($"{MyGlobals.OutputDir}/output.tsv", $"{_sortedSampledElements.Count}\t{candidate.Result}\t{MyGlobals.Stopwatch.Elapsed}\t{commandLineStr}\r\n");
            }

            public (Auto.ObjectModel.Pipeline[], ITransformer bestModel) InferPipelines(int numTransformLevels, int batchSize, int numOfTrainingRows)
            {
                // get available learners
                var learners = RecipeInference.AllowedLearners(_mlContext, TrainerKind, _maxNumIterations);
                PipelineSuggester.UpdateTrainers(learners);

                // get available transforms
                var transforms = InferTransforms();
                PipelineSuggester.UpdateTransforms(transforms);

                MainLearningLoop(batchSize);

                // temporary hack: retrain best model
                var bestModel = _sortedSampledElements.First().Value.TrainTransformer(_trainData);

                // return pipelines
                return (_sortedSampledElements.Values.Select(p => p.ToObjectModel()).ToArray(), bestModel);
            }
            
            private IEnumerable<SuggestedTransform> InferTransforms()
            {
                var data = _trainData;
                var args = new TransformInference.Arguments
                {
                    EstimatedSampleFraction = 1.0,
                    ExcludeFeaturesConcatTransforms = true
                };
                return TransformInference.InferTransforms(_mlContext, data, args);
            }
        }
    }
}