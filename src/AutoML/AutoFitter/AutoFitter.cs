// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class AutoFitter
    {
        private readonly IDebugLogger _debugLogger;
        private readonly IList<PipelineRunResult> _history;
        private readonly string _label;
        private readonly MLContext _mlContext;
        private readonly OptimizingMetricInfo _optimizingMetricInfo;
        private readonly IDictionary<string, ColumnPurpose> _purposeOverrides;
        private readonly AutoFitSettings _settings;
        private readonly IDataView _trainData;
        private readonly TaskKind _task;
        private readonly IDataView _validationData;

        public AutoFitter(MLContext mlContext, OptimizingMetricInfo metricInfo, AutoFitSettings settings, 
            TaskKind task, string label, IDataView trainData, IDataView validationData,
            IDictionary<string, ColumnPurpose> purposeOverrides, IDebugLogger debugLogger)
        {
            _debugLogger = debugLogger;
            _history = new List<PipelineRunResult>();
            _label = label;
            _mlContext = mlContext;
            _optimizingMetricInfo = metricInfo;
            _settings = settings ?? new AutoFitSettings();
            _purposeOverrides = purposeOverrides;
            _trainData = trainData;
            _task = task;
            _validationData = validationData;
        }

        public PipelineRunResult[] Fit()
        {
            IteratePipelinesAndFit();
            return _history.ToArray();
        }

        private void IteratePipelinesAndFit()
        {
            var stopwatch = Stopwatch.StartNew();
            var transforms = TransformInferenceApi.InferTransforms(_mlContext, _trainData, _label, _purposeOverrides);
            var availableTrainers = RecipeInference.AllowedTrainers(_mlContext, _task, _settings.StoppingCriteria.MaxIterations);

            do
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

            } while (_history.Count < _settings.StoppingCriteria.MaxIterations &&
                    stopwatch.Elapsed.TotalMinutes < _settings.StoppingCriteria.TimeOutInMinutes);
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
                WriteDebugLog(DebugStream.RunResult, $"{_history.Count}\t{runResult.Score}\t{stopwatch.Elapsed}\t{commandLineStr}");
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