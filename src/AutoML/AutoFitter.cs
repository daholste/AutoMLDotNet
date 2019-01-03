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
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class AutoFitter
    {
        private readonly IList<PipelineRunResult> _history;
        private readonly int _targetMaxNumIterations;
        private readonly MLContext _mlContext;
        private readonly OptimizingMetricInfo _optimizingMetricInfo;
        private readonly IterationBasedTerminator _terminator;
        private readonly IDataView _trainData;
        private readonly TaskKind _task;
        private readonly IDataView _validationData;

        public AutoFitter(MLContext mlContext, OptimizingMetricInfo metricInfo, IterationBasedTerminator terminator, 
            TaskKind task, int targetMaxNumIterations,
            IDataView trainData, IDataView validationData)
        {
            _history = new List<PipelineRunResult>();
            _targetMaxNumIterations = targetMaxNumIterations;
            _mlContext = mlContext;
            _optimizingMetricInfo = metricInfo;
            _terminator = terminator;
            _trainData = trainData;
            _task = task;
            _validationData = validationData;
        }

        public PipelineRunResult[] InferPipelines(int numTransformLevels, int batchSize, int numOfTrainingRows)
        {
            var availableTrainers = RecipeInference.AllowedTrainers(_mlContext, _task, _targetMaxNumIterations);
            var availableTransforms = InferTransforms();
            var pipelineSuggester = new RocketPipelineSuggester(_mlContext, _optimizingMetricInfo.IsMaximizing,
                availableTrainers, availableTransforms);

            MainLearningLoop(pipelineSuggester, batchSize);

            return _history.ToArray();
        }

        private void MainLearningLoop(IPipelineSuggester pipelineSuggester, int batchSize)
        {
            while (!_terminator.ShouldTerminate(_history.Count))
            {
                try
                {
                    // get next set of candidates
                    var currentBatchSize = batchSize;
                    if (_terminator is IterationBasedTerminator itr)
                    {
                        currentBatchSize = Math.Min(itr.RemainingIterations(_history.Count), batchSize);
                    }
                    var pipelines = pipelineSuggester.GetNextPipelines(_history, currentBatchSize);

                    // break if no candidates returned, means no valid pipeline available
                    if (!pipelines.Any())
                    {
                        break;
                    }

                    // evaluate candidates
                    foreach (var pipeline in pipelines)
                    {
                        try
                        {
                            ProcessPipeline(pipeline);
                        }
                        catch (Exception e)
                        {
                            File.AppendAllText($"{MyGlobals.OutputDir}/crash_dump1.txt", $"{pipeline.Trainer} Crashed {e}\r\n");
                            pipelineSuggester.MarkPipelineAsFailed(pipeline);
                        }
                    }
                }
                catch (Exception e)
                {
                    File.AppendAllText($"{MyGlobals.OutputDir}/crash_dump2.txt", $"{e}\r\n");
                }
            }
        }

        private void ProcessPipeline(Pipeline pipeline)
        {
            // run pipeline
            var stopwatch = Stopwatch.StartNew();
            var pipelineModel = pipeline.TrainTransformer(_trainData);
            var scoredValidationData = pipelineModel.Transform(_validationData);
            var evaluatedMetrics = GetEvaluatedMetrics(scoredValidationData);
            var score = GetPipelineScore(evaluatedMetrics);

            // save pipeline run
            var runResult = new PipelineRunResult(evaluatedMetrics, pipelineModel, pipeline, score, scoredValidationData);
            _history.Add(runResult);

            stopwatch.Stop();

            var transformsSb = new StringBuilder();
            foreach (var transform in pipeline.Transforms)
            {
                transformsSb.Append("xf=");
                transformsSb.Append(transform);
                transformsSb.Append(" ");
            }
            var commandLineStr = $"{transformsSb.ToString()} tr={pipeline.Trainer}";
            File.AppendAllText($"{MyGlobals.OutputDir}/output.tsv", $"{_history.Count}\t{score}\t{stopwatch.Elapsed}\t{commandLineStr}\r\n");
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

        private object GetEvaluatedMetrics(IDataView scoredData)
        {
            switch(_task)
            {
                case TaskKind.BinaryClassification:
                    return _mlContext.BinaryClassification.EvaluateNonCalibrated(scoredData);
                case TaskKind.MulticlassClassification:
                    return _mlContext.MulticlassClassification.Evaluate(scoredData);
                case TaskKind.Regression:
                    return _mlContext.Regression.Evaluate(scoredData);
                default:
                    throw new NotSupportedException("unsupported task type");
            }
        }

        private double GetPipelineScore(object evaluatedMetrics)
        {
            var type = evaluatedMetrics.GetType();
            if(type == typeof(BinaryClassificationMetrics))
            {
                return ((BinaryClassificationMetrics)evaluatedMetrics).Accuracy;
            }
            if (type == typeof(MultiClassClassifierMetrics))
            {
                return ((MultiClassClassifierMetrics)evaluatedMetrics).AccuracyMicro;
            }
            if (type == typeof(RegressionMetrics))
            {
                return ((RegressionMetrics)evaluatedMetrics).RSquared;
            }
            throw new NotSupportedException("unsupported task type");
        }
    }
}