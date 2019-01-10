using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public static class RegressionExtensions
    {
        public static RegressionResult AutoFit(this RegressionContext context,
            IDataView trainData, 
            string label, 
            IDataView validationData = null, 
            AutoFitSettings settings = null,
            InferredColumn[] inferredColumns = null,
            CancellationToken cancellationToken = default, 
            IProgress<RegressionIterationResult> iterationCallback = null)
        {
            return AutoFit(context, trainData, label, validationData, inferredColumns, settings,
                cancellationToken, iterationCallback, null);
        }

        // todo: instead of internal methods, use static debug class w/ singleton logger?
        internal static RegressionResult AutoFit(this RegressionContext context, 
            IDataView trainData, 
            string label, 
            IDataView validationData = null, 
            InferredColumn[] inferredColumns = null, 
            AutoFitSettings settings = null,
            CancellationToken cancellationToken = default, 
            IProgress<RegressionIterationResult> iterationCallback = null, 
            IDebugLogger debugLogger = null)
        {
            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.Fit(trainData, validationData, label, inferredColumns,
                settings, TaskKind.Regression, OptimizingMetric.RSquared, debugLogger);

            var results = new RegressionIterationResult[allPipelines.Length];
            for (var i = 0; i < results.Length; i++)
            {
                var iterationResult = allPipelines[i];
                var result = new RegressionIterationResult((RegressionMetrics)iterationResult.EvaluatedMetrics, iterationResult.Model, iterationResult.ScoredValidationData);
                results[i] = result;
            }
            var bestResult = new RegressionIterationResult((RegressionMetrics)bestPipeline.EvaluatedMetrics, bestPipeline.Model, bestPipeline.ScoredValidationData);
            return new RegressionResult(bestResult, results);
        }

        public static Pipeline GetPipeline(this RegressionContext context, IDataView dataView, string label)
        {
            return PipelineSuggesterApi.GetPipeline(TaskKind.Regression, dataView, label);
        }
    }

    public static class BinaryClassificationExtensions
    {
        public static BinaryClassificationResult AutoFit(this BinaryClassificationContext context,
            IDataView trainData, 
            string label, 
            IDataView validationData = null,
            InferredColumn[] inferredColumns = null,
            AutoFitSettings settings = null,
            CancellationToken cancellationToken = default, 
            IProgress<BinaryClassificationItertionResult> iterationCallback = null)
        {
            return AutoFit(context, trainData, label, validationData, inferredColumns, settings,
                cancellationToken, iterationCallback, null);
        }

        internal static BinaryClassificationResult AutoFit(this BinaryClassificationContext context,
            IDataView trainData, 
            string label, 
            IDataView validationData = null,
            InferredColumn[] inferredColumns = null,
            AutoFitSettings settings = null,
            CancellationToken cancellationToken = default,
            IProgress<BinaryClassificationItertionResult> iterationCallback = null, 
            IDebugLogger debugLogger = null)
        {
            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.Fit(trainData, validationData, label, inferredColumns,
                settings, TaskKind.BinaryClassification, OptimizingMetric.Accuracy,
                debugLogger);

            var results = new BinaryClassificationItertionResult[allPipelines.Length];
            for(var i = 0; i < results.Length; i++)
            {
                var iterationResult = allPipelines[i];
                var result = new BinaryClassificationItertionResult((BinaryClassificationMetrics)iterationResult.EvaluatedMetrics, iterationResult.Model, iterationResult.ScoredValidationData);
                results[i] = result;
            }
            var bestResult = new BinaryClassificationItertionResult((BinaryClassificationMetrics)bestPipeline.EvaluatedMetrics, bestPipeline.Model, bestPipeline.ScoredValidationData);
            return new BinaryClassificationResult(bestResult, results);
        }

        public static Pipeline GetPipeline(this BinaryClassificationContext context, IDataView dataView, string label)
        {
            return PipelineSuggesterApi.GetPipeline(TaskKind.BinaryClassification, dataView, label);
        }
    }

    public static class MulticlassExtensions
    {
        public static MulticlassClassificationResult AutoFit(this MulticlassClassificationContext context,
            IDataView trainData, 
            string label, 
            IDataView validationData = null,
            InferredColumn[] inferredColumns = null,
            AutoFitSettings settings = null,
            CancellationToken cancellationToken = default, 
            IProgress<MulticlassClassificationIterationResult> iterationCallback = null)
        {
            return AutoFit(context, trainData, label, validationData, inferredColumns, settings,
                cancellationToken, iterationCallback, null);
        }

        internal static MulticlassClassificationResult AutoFit(this MulticlassClassificationContext context,
            IDataView trainData, 
            string label, 
            IDataView validationData = null,
            InferredColumn[] inferredColumns = null, 
            AutoFitSettings settings = null,
            CancellationToken cancellationToken = default,
            IProgress<MulticlassClassificationIterationResult> iterationCallback = null, IDebugLogger debugLogger = null)
        {
            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.Fit(trainData, validationData, label, inferredColumns,
                settings, TaskKind.MulticlassClassification, OptimizingMetric.Accuracy, debugLogger);

            var results = new MulticlassClassificationIterationResult[allPipelines.Length];
            for (var i = 0; i < results.Length; i++)
            {
                var iterationResult = allPipelines[i];
                var result = new MulticlassClassificationIterationResult((MultiClassClassifierMetrics)iterationResult.EvaluatedMetrics, iterationResult.Model, iterationResult.ScoredValidationData);
                results[i] = result;
            }
            var bestResult = new MulticlassClassificationIterationResult((MultiClassClassifierMetrics)bestPipeline.EvaluatedMetrics, bestPipeline.Model, bestPipeline.ScoredValidationData);
            return new MulticlassClassificationResult(bestResult, results);
        }

        public static Pipeline GetPipeline(this MulticlassClassificationContext context, IDataView dataView, string label)
        {
            return PipelineSuggesterApi.GetPipeline(TaskKind.MulticlassClassification, dataView, label);
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
            return ColumnInferenceApi.InferColumns(mlContext, path, label, hasHeader, separator);
        }

        // Auto reader (includes column inference)
        public static IDataView AutoRead(this DataOperations catalog, Stream stream)
        {
            throw new NotImplementedException();
        }

        public static IDataView AutoRead(this DataOperations catalog, string path, string label, bool hasHeader = false, string separator = null)
        {
            var mlContext = new MLContext();
            var columnInferenceResult = ColumnInferenceApi.InferColumns(mlContext, path, label, hasHeader, separator);
            var textLoader = columnInferenceResult.BuildTextLoader();
            return textLoader.Read(path);
        }

        public static IDataView AutoRead(this DataOperations catalog, IMultiStreamSource source, string label, bool hasHeader = false, string separator = null)
        {
            var mlContext = new MLContext();
            var columnInferenceResult = ColumnInferenceApi.InferColumns(mlContext, source, label, hasHeader, separator);
            var textLoader = columnInferenceResult.BuildTextLoader();
            return textLoader.Read(source);
        }

        public static TextLoader CreateTextReader(this DataOperations catalog, ColumnInferenceResult columnInferenceResult)
        {
            return columnInferenceResult.BuildTextLoader();
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

        internal TextLoader BuildTextLoader()
        {
            var context = new MLContext();
            return new TextLoader(context, new TextLoader.Arguments() {
                AllowQuoting = IsQuoted,
                AllowSparse = IsSparse,
                Column = InferredColumns.Select(c => c.ToTextLoaderColumn()).ToArray(),
                Separator = Separator
            });
        }
    }

    public class InferredColumn
    {
        public string Name;
        public DataKind Type;
        public TextLoader.Range[] Source;

        // todo: have an internal copy of ColumnPurpose?
        public ColumnPurpose ColumnPurpose;

        public InferredColumn(string name, DataKind type, TextLoader.Range[] source, ColumnPurpose columnPurpose)
        {
            Name = name;
            Type = type;
            Source = source;
            ColumnPurpose = columnPurpose;
        }

        internal TextLoader.Column ToTextLoaderColumn()
        {
            return new TextLoader.Column()
            {
                Name = Name,
                Type = Type,
                Source = Source
            };
        }

        internal PurposeInference.Column[] ToInternalColumnPurposes()
        {
            var columnIndexList = AutoMlUtils.GetColumnIndexList(Source);
            var result = new PurposeInference.Column[columnIndexList.Count];
            for(var i = 0; i < columnIndexList.Count; i++)
            {
                var internalColumn = new PurposeInference.Column(columnIndexList[i], ColumnPurpose, Type);
                result[i] = internalColumn;
            }
            return result;
        }
    }

    public class AutoFitSettings
    {
        public ExperimentStoppingCriteria StoppingCriteria = new ExperimentStoppingCriteria();
        internal IterationStoppingCriteria IterationStoppingCriteria;
        internal Concurrency Concurrency;
        internal Filters Filters;
        internal CrossValidationSettings CrossValidationSettings;
        internal OptimizingMetric OptimizingMetric;
        internal bool EnableEnsembling;
        internal bool EnableModelExplainability;
        internal bool EnableAutoTransformation;

        // spec question: Are following automatic or a user setting?
        internal bool EnableSubSampling;
        internal bool EnableCaching;
        internal bool ExternalizeTraining;
        internal TraceLevel TraceLevel; // Should this be controlled through code or appconfig?
    }

    public class ExperimentStoppingCriteria
    {
        public int MaxIterations = 100;
        public int TimeOutInMinutes = 300;
        internal bool StopAfterConverging;
        internal double ExperimentExitScore;
    }

    internal class Filters
    {
        internal IEnumerable<Trainers> WhitelistTrainers;
        internal IEnumerable<Trainers> BlackListTrainers;
        internal IEnumerable<Transformers> WhitelistTransformers;
        internal IEnumerable<Transformers> BlacklistTransformers;
        internal bool PreferExplainability;
        internal bool PreferInferenceSpeed;
        internal bool PreferSmallDeploymentSize;
        internal bool PreferSmallMemoryFootprint;
    }

    public class IterationStoppingCriteria
    {
        internal int TimeOutInSeconds;
        internal bool TerminateOnLowAccuracy;
    }

    public class Concurrency
    {
        internal int MaxConcurrentIterations;
        internal int MaxCoresPerIteration;
    }

    internal enum Trainers
    {
    }

    internal enum Transformers
    {
    }

    internal class CrossValidationSettings
    {
        internal int NumberOfFolds;
        internal int ValidationSizePercentage;
        internal IEnumerable<string> StratificationColumnNames;
    }

    public class BinaryClassificationResult
    {
        public readonly BinaryClassificationItertionResult BestPipeline;
        public readonly BinaryClassificationItertionResult[] IterationResults;

        public BinaryClassificationResult(BinaryClassificationItertionResult bestPipeline,
            BinaryClassificationItertionResult[] iterationResults)
        {
            BestPipeline = bestPipeline;
            IterationResults = iterationResults;
        }
    }

    public class MulticlassClassificationResult
    {
        public readonly MulticlassClassificationIterationResult BestPipeline;
        public readonly MulticlassClassificationIterationResult[] IterationResults;

        public MulticlassClassificationResult(MulticlassClassificationIterationResult bestPipeline,
            MulticlassClassificationIterationResult[] iterationResults)
        {
            BestPipeline = bestPipeline;
            IterationResults = iterationResults;
        }
    }

    public class RegressionResult
    {
        public readonly RegressionIterationResult BestPipeline;
        public readonly RegressionIterationResult[] IterationResults;

        public RegressionResult(RegressionIterationResult bestPipeline,
            RegressionIterationResult[] iterationResults)
        {
            BestPipeline = bestPipeline;
            IterationResults = iterationResults;
        }
    }

    public class BinaryClassificationItertionResult : IterationResult
    {
        public readonly BinaryClassificationMetrics Metrics;

        public BinaryClassificationItertionResult(BinaryClassificationMetrics metrics,
            ITransformer model, IDataView scoredValidationData) : base(model, scoredValidationData)
        {
            Metrics = metrics;
        }
    }

    public class MulticlassClassificationIterationResult : IterationResult
    {
        public readonly MultiClassClassifierMetrics Metrics;

        public MulticlassClassificationIterationResult(MultiClassClassifierMetrics metrics,
            ITransformer model, IDataView scoredValidationData) : base(model, scoredValidationData)
        {
            Metrics = metrics;
        }
    }

    public class RegressionIterationResult : IterationResult
    {
        public readonly RegressionMetrics Metrics;

        public RegressionIterationResult(RegressionMetrics metrics,
            ITransformer model, IDataView scoredValidationData) : base(model, scoredValidationData)
        {
            Metrics = metrics;
        }
    }

    public class IterationResult
    {
        public readonly ITransformer Model;
        public readonly IDataView ScoredValidationData;
        internal readonly Pipeline Pipeline;

        public IterationResult(ITransformer model, IDataView scoredValidationData, Pipeline pipeline = null)
        {
            Model = model;
            ScoredValidationData = scoredValidationData;
            Pipeline = pipeline;
        }
    }

    public enum InferenceType
    {
        Seperator,
        Header,
        Label,
        Task,
        ColumnDataKind,
        ColumnPurpose,
        Tranform,
        Trainer,
        Hyperparams,
        ColumnSplit
    }

    public class InferenceException : Exception
    {
        public InferenceType InferenceType;

        public InferenceException(InferenceType inferenceType, string message)
        : base(message)
        {
        }

        public InferenceException(InferenceType inferenceType, string message, Exception inner)
            : base(message, inner)
        {
        }
    }

    public class Pipeline
    {
        public readonly PipelineNode[] Elements;

        public Pipeline(PipelineNode[] elements)
        {
            Elements = elements;
        }
    }

    public class PipelineNode
    {
        public readonly string Name;
        public readonly PipelineNodeType ElementType;
        public readonly string[] InColumns;
        public readonly string[] OutColumns;
        public readonly IDictionary<string, object> Properties;

        public PipelineNode(string name, PipelineNodeType elementType,
            string[] inColumns, string[] outColumns,
            IDictionary<string, object> properties)
        {
            Name = name;
            ElementType = elementType;
            InColumns = inColumns;
            OutColumns = outColumns;
            Properties = properties;
        }
    }

    public enum PipelineNodeType
    {
        Transform,
        Trainer
    }
}
