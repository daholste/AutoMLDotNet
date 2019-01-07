using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Auto.Public
{
    public static class BinaryClassificationPipelineSuggester
    {
        public static Pipeline[] GetNextPipelines(Pipeline[] history, RegressionMetrics[] metrics, IDataView trainData, string label, List<string> whiteListeTrainers = null, List<string> blockListTrainers = null)
        {
            throw new NotImplementedException();
        }

        public static Pipeline GetFirstPipeline(IDataView dataView, string label)
        {
            var mlContext = new MLContext();
            var availableTransforms = TransformInferenceApi.InferTransforms(mlContext, dataView, label);
            var availableTrainers = RecipeInference.AllowedTrainers(mlContext, TaskKind.BinaryClassification, 1);
            var pipeline = new Auto.Pipeline(availableTransforms, availableTrainers.First(), mlContext);
            return pipeline.ToObjectModel();
        }
    }

    public static class RegressionExtensions
    {
        public static RegressionResult AutoFit(this RegressionContext context,
            IDataView trainData, string label, IDataView validationData = null, IEstimator<ITransformer> preprocessor = null, AutoFitSettings settings = null)
        {
            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.AutoFit(trainData, validationData, label, settings.MaxIterations, 
                preprocessor, TaskKind.Regression, OptimizingMetric.RSquared);

            var results = new RegressionPipelineResult[allPipelines.Length];
            for (var i = 0; i < results.Length; i++)
            {
                var pipelineResult = allPipelines[i];
                var result = new RegressionPipelineResult((RegressionMetrics)pipelineResult.EvaluatedMetrics, pipelineResult.Model, pipelineResult.ScoredValidationData);
                results[i] = result;
            }
            var bestResult = new RegressionPipelineResult((RegressionMetrics)bestPipeline.EvaluatedMetrics, bestPipeline.Model, bestPipeline.ScoredValidationData);
            return new RegressionResult(bestResult, results);
        }
    }

    public static class BinaryClassificationExtensions
    {
        public static BinaryClassificationResult AutoFit(this BinaryClassificationContext context,
            IDataView trainData, string label, IDataView validationData = null, IEstimator<ITransformer> preprocessor = null, AutoFitSettings settings = null)
        {
            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.AutoFit(trainData, validationData, label, settings.MaxIterations, 
                preprocessor, TaskKind.BinaryClassification, OptimizingMetric.Accuracy);

            var results = new BinaryClassificationPipelineResult[allPipelines.Length];
            for(var i = 0; i < results.Length; i++)
            {
                var pipelineResult = allPipelines[i];
                var result = new BinaryClassificationPipelineResult((BinaryClassificationMetrics)pipelineResult.EvaluatedMetrics, pipelineResult.Model, pipelineResult.ScoredValidationData);
                results[i] = result;
            }
            var bestResult = new BinaryClassificationPipelineResult((BinaryClassificationMetrics)bestPipeline.EvaluatedMetrics, bestPipeline.Model, bestPipeline.ScoredValidationData);
            return new BinaryClassificationResult(bestResult, results);
        }
    }

    public static class MulticlassExtensions
    {
        public static MulticlassClassificationResult AutoFit(this MulticlassClassificationContext context,
            IDataView trainData, string label, IDataView validationData = null, IEstimator<ITransformer> preprocessor = null, AutoFitSettings settings = null)
        {
            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.AutoFit(trainData, validationData, label, settings.MaxIterations,
                preprocessor, TaskKind.MulticlassClassification, OptimizingMetric.Accuracy);

            var results = new MulticlassClassificationPipelineResult[allPipelines.Length];
            for (var i = 0; i < results.Length; i++)
            {
                var pipelineResult = allPipelines[i];
                var result = new MulticlassClassificationPipelineResult((MultiClassClassifierMetrics)pipelineResult.EvaluatedMetrics, pipelineResult.Model, pipelineResult.ScoredValidationData);
                results[i] = result;
            }
            var bestResult = new MulticlassClassificationPipelineResult((MultiClassClassifierMetrics)bestPipeline.EvaluatedMetrics, bestPipeline.Model, bestPipeline.ScoredValidationData);
            return new MulticlassClassificationResult(bestResult, results);
        }
    }

    public static class TransformExtensions
    {
        public static IEstimator<ITransformer> InferTransforms(this TransformsCatalog catalog, IDataView data, string label)
        {
            var mlContext = new MLContext();
            var suggestedTransforms = TransformInferenceApi.InferTransforms(mlContext, data, label);
            var estimators = suggestedTransforms.Select(s => s.Estimator);
            var pipeline = new EstimatorChain<ITransformer>();
            foreach(var estimator in estimators)
            {
                pipeline = pipeline.Append(estimator);
            }
            return pipeline;
        }
    }

    public static class DataExtensions
    {
        // Delimiter, header, column datatype inference
        public static ColumnInferenceResult InferColumns(this DataOperations catalog, string path, string label, bool hasHeader = false, string separator = null, TextLoader.Column[] columns = null)
        {
            // todo: respect & test column overrides param
            var mlContext = new MLContext();
            return SchemaInferenceApi.InferColumns(mlContext, path, label, hasHeader, separator);
        }

        // Auto reader (includes column inference)
        public static IDataView AutoRead(this DataOperations catalog, Stream stream)
        {
            throw new NotImplementedException();
        }

        public static IDataView AutoRead(this DataOperations catalog, string path, string label, bool hasHeader = false, string separator = null)
        {
            var mlContext = new MLContext();
            var columnInferenceResult = SchemaInferenceApi.InferColumns(mlContext, path, label, hasHeader, separator);
            var textLoader = columnInferenceResult.BuildTextLoader(mlContext);
            return textLoader.Read(path);
        }

        public static IDataView AutoRead(this DataOperations catalog, IMultiStreamSource source, string label, bool hasHeader = false, string separator = null)
        {
            var mlContext = new MLContext();
            var columnInferenceResult = SchemaInferenceApi.InferColumns(mlContext, source, label, hasHeader, separator);
            var textLoader = columnInferenceResult.BuildTextLoader(mlContext);
            return textLoader.Read(source);
        }

        // Task inference
        public static MachineLearningTaskType InferTask(this DataOperations catalog, IDataView dataView)
        {
            throw new NotImplementedException();
        }

        public enum MachineLearningTaskType
        {
            Regression,
            BinaryClassification,
            MultiClassClassification
        }
    }
    
    public class ColumnInferenceResult
    {
        public readonly bool IsQuoted;
        public readonly bool IsSparse;
        public readonly InferredColumn[] InferredColumns;
        public readonly string Separator;

        public ColumnInferenceResult(bool isQuoted, bool isSparse, InferredColumn[] inferredColumns,
            string separator)
        {
            IsQuoted = isQuoted;
            IsSparse = isSparse;
            InferredColumns = inferredColumns;
            Separator = separator;
        }

        // todo: should we keep public, or make this private?
        public TextLoader BuildTextLoader(MLContext context)
        {
            return new TextLoader(context, new TextLoader.Arguments() {
                AllowQuoting = IsQuoted,
                AllowSparse = IsSparse,
                Column = InferredColumns.Select(c => c.TextLoaderColumn).ToArray(),
                Separator = Separator
            });
        }
    }

    public class InferredColumn
    {
        public readonly TextLoader.Column TextLoaderColumn;

        // todo: have an internal copy of ColumnPurpose?
        public readonly ColumnPurpose ColumnPurpose;

        public InferredColumn(TextLoader.Column textLoaderColumn, ColumnPurpose columnPurpose)
        {
            TextLoaderColumn = textLoaderColumn;
            ColumnPurpose = columnPurpose;
        }
    }

    public class RunHistory
    {
        
    }

    public class AutoFitSettings
    {
        public StoppingCriteria StoppingCriteria;
        public int MaxIterations;
    }

    public class StoppingCriteria
    {
        public int MaxIterations;
        public int MaxSeconds;
        public bool StopAfterConverging;
        public double ExitScore;
    }

    public class BinaryClassificationResult
    {
        public readonly BinaryClassificationPipelineResult BestPipeline;
        public readonly BinaryClassificationPipelineResult[] PipelineResults;

        public BinaryClassificationResult(BinaryClassificationPipelineResult bestPipeline,
            BinaryClassificationPipelineResult[] pipelineResults)
        {
            BestPipeline = bestPipeline;
            PipelineResults = pipelineResults;
        }
    }

    public class MulticlassClassificationResult
    {
        public readonly MulticlassClassificationPipelineResult BestPipeline;
        public readonly MulticlassClassificationPipelineResult[] PipelineResults;

        public MulticlassClassificationResult(MulticlassClassificationPipelineResult bestPipeline,
            MulticlassClassificationPipelineResult[] pipelineResults)
        {
            BestPipeline = bestPipeline;
            PipelineResults = pipelineResults;
        }
    }

    public class RegressionResult
    {
        public readonly RegressionPipelineResult BestPipeline;
        public readonly RegressionPipelineResult[] PipelineResults;

        public RegressionResult(RegressionPipelineResult bestPipeline,
            RegressionPipelineResult[] pipelineResults)
        {
            BestPipeline = bestPipeline;
            PipelineResults = pipelineResults;
        }
    }

    public class BinaryClassificationPipelineResult : PipelineResult
    {
        public readonly BinaryClassificationMetrics Metrics;

        public BinaryClassificationPipelineResult(BinaryClassificationMetrics metrics,
            ITransformer model, IDataView scoredValidationData) : base(model, scoredValidationData)
        {
            Metrics = metrics;
        }
    }

    public class MulticlassClassificationPipelineResult : PipelineResult
    {
        public readonly MultiClassClassifierMetrics Metrics;

        public MulticlassClassificationPipelineResult(MultiClassClassifierMetrics metrics,
            ITransformer model, IDataView scoredValidationData) : base(model, scoredValidationData)
        {
            Metrics = metrics;
        }
    }

    public class RegressionPipelineResult : PipelineResult
    {
        public readonly RegressionMetrics Metrics;

        public RegressionPipelineResult(RegressionMetrics metrics,
            ITransformer model, IDataView scoredValidationData) : base(model, scoredValidationData)
        {
            Metrics = metrics;
        }
    }

    public class PipelineResult
    {
        public readonly ITransformer Model;
        public readonly IDataView ScoredValidationData;

        public PipelineResult(ITransformer model, IDataView scoredValidationData)
        {
            Model = model;
            ScoredValidationData = scoredValidationData;
        }
    }
}
