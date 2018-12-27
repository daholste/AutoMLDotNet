// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Data;
using System.IO;
using System.Text;
using Microsoft.ML.PipelineInference2;
using MacroUtils = Microsoft.ML.PipelineInference2.MacroUtils;

namespace Microsoft.ML.PipelineInference2
{
    /// <summary>
    /// Class for generating potential recipes/pipelines, testing them, and zeroing in on the best ones.
    /// For now, only works with maximizing metrics (AUC, Accuracy, etc.).
    /// </summary>
    public class AutoInference
    {
        public struct ColumnInfo
        {
            public string Name { get; set; }
            public ColumnType ItemType { get; set; }
            public bool IsHidden { get; set; }
            public override string ToString() => Name;
        }

        public sealed class ReversedComparer<T> : IComparer<T>
        {
            public int Compare(T x, T y)
            {
                return Comparer<T>.Default.Compare(y, x);
            }
        }

        /// <summary>
        /// Alias to refer to this construct by an easy name.
        /// </summary>
        public class LevelDependencyMap : Dictionary<ColumnInfo, List<TransformInference.SuggestedTransform>> { }

        /// <summary>
        /// Alias to refer to this construct by an easy name.
        /// </summary>
        public class DependencyMap : Dictionary<int, LevelDependencyMap> { }

        /// <summary>
        /// Class that holds state for an autoML search-in-progress. Should be able to resume search, given this object.
        /// </summary>
        public sealed class AutoMlMlState
        {
            private readonly SortedList<double, PipelinePattern> _sortedSampledElements;
            private readonly List<PipelinePattern> _history;
            private readonly MLContext _env;
            private IDataView _trainData;
            private IDataView _testData;
            private IDataView _transformedData;
            private ITerminator _terminator;
            private int _maxNumIterations;
            private string[] _requestedLearners;
            private TransformInference.SuggestedTransform[] _availableTransforms;
            private RecipeInference.SuggestedRecipe.SuggestedLearner[] _availableLearners;
            private DependencyMap _dependencyMapping;
            public IPipelineOptimizer AutoMlEngine { get; set; }
            public PipelinePattern[] BatchCandidates { get; set; }
            public SupportedMetric Metric { get; }
            public MacroUtils.TrainerKinds TrainerKind { get; }

            public AutoMlMlState(MLContext env, SupportedMetric metric, IPipelineOptimizer autoMlEngine,
                ITerminator terminator, MacroUtils.TrainerKinds trainerKind, int maxNumIterations,
                IDataView trainData = null, IDataView testData = null,
                string[] requestedLearners = null)
            {
                //Contracts.CheckValue(env, nameof(env));
                _sortedSampledElements =
                    metric.IsMaximizing ? new SortedList<double, PipelinePattern>(new ReversedComparer<double>()) :
                        new SortedList<double, PipelinePattern>();
                _history = new List<PipelinePattern>();
                _env = env;
                //_host = _env.Register("AutoMlState");
                //_ch = _host.Start("AutoMlStateChannel");
                _trainData = trainData;
                _testData = testData;
                _terminator = terminator;
                _maxNumIterations = maxNumIterations;
                _requestedLearners = requestedLearners;
                AutoMlEngine = autoMlEngine;
                BatchCandidates = new PipelinePattern[] { };
                Metric = metric;
                TrainerKind = trainerKind;
            }

            private void MainLearningLoop(int batchSize, int numOfTrainingRows)
            {
                var overallExecutionTime = Stopwatch.StartNew();
                var stopwatch = new Stopwatch();
                var probabilityUtils = new SweeperProbabilityUtils();

                while (!_terminator.ShouldTerminate(_history))
                {
                    try
                    {
                        // Get next set of candidates
                        var currentBatchSize = batchSize;
                        if (_terminator is IterationTerminator itr)
                            currentBatchSize = Math.Min(itr.RemainingIterations(_history), batchSize);
                        var candidates = AutoMlEngine.GetNextCandidates(_sortedSampledElements.Values, currentBatchSize);

                        // Break if no candidates returned, means no valid pipeline available.
                        if (candidates.Length == 0)
                            break;

                        // Evaluate them on subset of data
                        foreach (var candidate in candidates)
                        {
                            try
                            {
                                ProcessPipeline(probabilityUtils, stopwatch, candidate, numOfTrainingRows);
                            }
                            catch (Exception e)
                            {
                                File.AppendAllText($"{MyGlobals.OutputDir}/crash_dump1.txt", $"{candidate.Learner.PipelineNode} Crashed {e}\r\n");
                                MyGlobals.FailedPipelineHashes.Add(candidate.Learner.PipelineNode.ToString());
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

            private void ProcessPipeline(SweeperProbabilityUtils utils, Stopwatch stopwatch, PipelinePattern candidate, int numOfTrainingRows)
            {
                // Create a randomized numer of rows to do train/test with.
                int randomizedNumberOfRows =
                    (int)Math.Floor(utils.NormalRVs(1, numOfTrainingRows, (double)numOfTrainingRows / 10).First());
                if (randomizedNumberOfRows > numOfTrainingRows)
                    randomizedNumberOfRows = numOfTrainingRows - (randomizedNumberOfRows - numOfTrainingRows);

                // Run pipeline, and time how long it takes
                stopwatch.Restart();
                candidate.RunTrainTestExperiment(_trainData,
                    _testData, Metric, TrainerKind, _env, out var testMetricVal);
                stopwatch.Stop();

                // Handle key collisions on sorted list
                while (_sortedSampledElements.ContainsKey(testMetricVal))
                    testMetricVal += 1e-10;

                // Save performance score
                candidate.PerformanceSummary = new PipelineSweeperRunSummary(testMetricVal, randomizedNumberOfRows, stopwatch.ElapsedMilliseconds, 0);
                _sortedSampledElements.Add(candidate.PerformanceSummary.MetricValue, candidate);
                _history.Add(candidate);

                var transformsSb = new StringBuilder();
                foreach (var transform in candidate.Transforms)
                {
                    transformsSb.Append("xf:");
                    transformsSb.Append(transform);
                    transformsSb.Append(" ");
                }
                var learnerStr = candidate.Learner.ToString();
                learnerStr = learnerStr.Replace(",", "");
                learnerStr = learnerStr.Replace("False", "-");
                learnerStr = learnerStr.Replace("True", "+");
                learnerStr = learnerStr.Replace("LearningRate:0 ", "");
                learnerStr = learnerStr.Replace("NumLeaves:0", "");
                learnerStr = learnerStr.Replace("Trainers.", "");
                learnerStr = learnerStr.Replace("LightGbmClassifier", "LightGBMMulticlass");
                learnerStr = learnerStr.Replace("LightGbmBinaryClassifier", "LightGBMBinary");
                learnerStr = learnerStr.Replace("LogisticRegressionClassifier", "MultiClassLogisticRegression");
                learnerStr = learnerStr.Replace("FastTreeBinaryClassifier", "FastTreeBinaryClassification");
                learnerStr = learnerStr.Replace("FieldAwareFactorizationMachineBinaryClassifier", "FieldAwareFactorizationMachine");
                var commandLineStr = $"{transformsSb.ToString()} tr={learnerStr}";
                File.AppendAllText($"{MyGlobals.OutputDir}/output.tsv", $"{_sortedSampledElements.Count}\t{candidate.PerformanceSummary.MetricValue}\t{MyGlobals.Stopwatch.Elapsed}\t{commandLineStr}\r\n");
            }

            private TransformInference.SuggestedTransform[] InferAndFilter(IDataView data, TransformInference.Arguments args)
            {
                // Infer transforms using experts
                var levelTransforms = TransformInference.InferTransforms(_env, data, args);
                return levelTransforms;
            }

            public void InferSearchSpace(int numTransformLevels)
            {
                var learners = RecipeInference.AllowedLearners(_env, TrainerKind, _maxNumIterations).ToArray();
                if (_requestedLearners != null && _requestedLearners.Length > 0)
                    learners = learners.Where(l => _requestedLearners.Contains(l.LearnerName)).ToArray();
                
                ComputeSearchSpace(numTransformLevels, learners, (b, c) => InferAndFilter(b, c));
            }

            public IEnumerable<PipelinePattern> InferPipelines(int numTransformLevels, int batchSize, int numOfTrainingRows)
            {
                //_env.AssertValue(_trainData, nameof(_trainData), "Must set training data prior to calling method.");
                //_env.AssertValue(_testData, nameof(_testData), "Must set test data prior to calling method.");

                //var h = _env.Register("InferPipelines");
                //using (var ch = h.Start("InferPipelines"))
                //{

                // Check if search space has not been initialized. If not,
                // run method to define it usign inference.
                if (!IsSearchSpaceDefined())
                    InferSearchSpace(numTransformLevels);

                // Learn for a given number of iterations
                MainLearningLoop(batchSize, numOfTrainingRows);

                // Return best pipeline seen
                return _sortedSampledElements.Values;

                //}
            }

            /// <summary>
            /// Search space is transforms X learners X hyperparameters.
            /// </summary>
            private void ComputeSearchSpace(int numTransformLevels, RecipeInference.SuggestedRecipe.SuggestedLearner[] learners,
                Func<IDataView, TransformInference.Arguments, TransformInference.SuggestedTransform[]> transformInferenceFunction)
            {
                //_env.AssertValue(_trainData, nameof(_trainData), "Must set training data prior to inferring search space.");

                //var h = _env.Register("ComputeSearchSpace");

                //using (var ch = h.Start("ComputeSearchSpace"))
                //{
                //_env.Check(IsValidLearnerSet(learners), "Unsupported learner encountered, cannot update search space.");

                var dataSample = _trainData;
                var inferenceArgs = new TransformInference.Arguments
                {
                    EstimatedSampleFraction = 1.0,
                    ExcludeFeaturesConcatTransforms = true
                };

                // Initialize structure for mapping columns back to specific transforms
                var dependencyMapping = new DependencyMap
                {
                    {0, AutoMlUtils.ComputeColumnResponsibilities(dataSample, new TransformInference.SuggestedTransform[0])}
                };

                // Get suggested transforms for all levels. Defines another part of search space.
                var transformsList = new List<TransformInference.SuggestedTransform>();
                for (int i = 0; i < numTransformLevels; i++)
                {
                    // Update level for transforms
                    inferenceArgs.Level = i + 1;

                    // Infer transforms using experts
                    var levelTransforms = transformInferenceFunction(dataSample, inferenceArgs);

                    // If no more transforms to apply, dataSample won't change. So end loop.
                    if (levelTransforms.Length == 0)
                        break;

                    // Make sure we don't overflow our bitmask
                    if (levelTransforms.Max(t => t.AtomicGroupId) > 64)
                        break;

                    // Level-up atomic group id offset.
                    inferenceArgs.AtomicIdOffset = levelTransforms.Max(t => t.AtomicGroupId) + 1;

                    // Apply transforms to dataview for this level.
                    dataSample = AutoMlUtils.ApplyTransformSet(dataSample, levelTransforms);

                    // Keep list of which transforms can be responsible for which output columns
                    dependencyMapping.Add(inferenceArgs.Level,
                        AutoMlUtils.ComputeColumnResponsibilities(dataSample, levelTransforms));
                    transformsList.AddRange(levelTransforms);
                }

                var transforms = transformsList.ToArray();

                // Save state, for resuming learning
                _availableTransforms = transforms;
                _availableLearners = learners;
                _dependencyMapping = dependencyMapping;
                _transformedData = dataSample;

                // Update autoML engine to know what the search space looks like
                AutoMlEngine.SetSpace(_availableTransforms, _availableLearners,
                    _trainData, _transformedData, _dependencyMapping, Metric.IsMaximizing);
                //}
            }

            public bool IsSearchSpaceDefined() => _availableLearners != null && _availableTransforms != null;
        }
    }
}
