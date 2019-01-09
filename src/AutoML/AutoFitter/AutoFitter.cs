// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Auto
{
    internal class AutoFitter
    {
        private readonly IDebugLogger _debugLogger;
        private readonly IList<PipelineRunResult> _history;
        private readonly string _label;
        private readonly int _maxIterations;
        private readonly MLContext _mlContext;
        private readonly OptimizingMetricInfo _optimizingMetricInfo;
        private readonly PurposeInference.Column[] _puproseOverrides;
        private readonly IterationBasedTerminator _terminator;
        private readonly IDataView _trainData;
        private readonly TaskKind _task;
        private readonly IDataView _validationData;

        public AutoFitter(MLContext mlContext, OptimizingMetricInfo metricInfo, IterationBasedTerminator terminator, 
            TaskKind task, int maxIterations, string label, PurposeInference.Column[] puproseOverrides,
            IDataView trainData, IDataView validationData, IDebugLogger debugLogger)
        {
            _debugLogger = debugLogger;
            _history = new List<PipelineRunResult>();
            _label = label;
            _maxIterations = maxIterations;
            _mlContext = mlContext;
            _optimizingMetricInfo = metricInfo;
            _puproseOverrides = puproseOverrides;
            _terminator = terminator;
            _trainData = trainData;
            _task = task;
            _validationData = validationData;
        }

        public PipelineRunResult[] InferPipelines(int batchSize)
        {
            MainLearningLoop();
            return _history.ToArray();
        }

        private void MainLearningLoop()
        {
            var transforms = TransformInferenceApi.InferTransforms(_mlContext, _trainData, _label, _puproseOverrides);
            var availableTrainers = RecipeInference.AllowedTrainers(_mlContext, _task, _maxIterations);

            while (!_terminator.ShouldTerminate(_history.Count))
            {
                try
                {
                    // get next pipeline
                    var pipeline = PipelineSuggester.GetNextPipeline(_history, transforms, availableTrainers, _optimizingMetricInfo.IsMaximizing);

                    // break if no candidates returned, means no valid pipeline available
                    if (pipeline == null)
                    {
                        break;
                    }

                    // evaluate pipeline
                    ProcessPipeline(pipeline);
                }
                catch (Exception ex)
                {
                    WriteDebugLog(DebugStream.Exception, $"{ex}");
                }
            }
        }

        private void ProcessPipeline(InferredPipeline pipeline)
        {
            // run pipeline
            var stopwatch = Stopwatch.StartNew();

            PipelineRunResult runResult;
            try
            {
                var pipelineModel = pipeline.TrainTransformer(_trainData);
                var scoredValidationData = pipelineModel.Transform(_validationData);
                var evaluatedMetrics = GetEvaluatedMetrics(scoredValidationData);
                var score = GetPipelineScore(evaluatedMetrics);
                runResult = new PipelineRunResult(evaluatedMetrics, pipelineModel, pipeline, score, scoredValidationData);
            }
            catch(Exception ex)
            {
                WriteDebugLog(DebugStream.Exception, $"{pipeline.Trainer} Crashed {ex}");
                runResult = new PipelineRunResult(pipeline, false);
            }

            // save pipeline run
            _history.Add(runResult);

            // debug log pipeline result
            if(runResult.RunSucceded)
            {
                var transformsSb = new StringBuilder();
                foreach (var transform in pipeline.Transforms)
                {
                    transformsSb.Append("xf=");
                    transformsSb.Append(transform);
                    transformsSb.Append(" ");
                }
                var commandLineStr = $"{transformsSb.ToString()} tr={pipeline.Trainer}";
                WriteDebugLog(DebugStream.RunResult, $"{_history.Count}\t{runResult.Score}\t{stopwatch.Elapsed}\t{commandLineStr}\r\n");
            }
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
                // should not be possible to reach here
                default:
                    throw new InvalidOperationException($"unsupported machine learning task type {_task}");
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
            
            // should not be possible to reach here
            throw new InvalidOperationException($"unsupported machine learning task type {_task}");
        }

        private void WriteDebugLog(DebugStream stream, string message)
        {
            if(_debugLogger == null)
            {
                return;
            }

            _debugLogger.Log(stream, message);
        }
    }
}