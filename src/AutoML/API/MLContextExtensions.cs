using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Auto
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
            var pipeline = new Auto.InferredPipeline(availableTransforms, availableTrainers.First(), mlContext);
            return pipeline.ToPipeline();
        }
    }

    public static class RegressionExtensions
    {
        public static RegressionResult AutoFit(this RegressionContext context,
            IDataView trainData, string label, IDataView validationData = null, IEstimator<ITransformer> preprocessor = null, AutoFitSettings settings = null,
            CancellationToken cancellationToken = default(CancellationToken), InferredColumn[] inferredColumns = null,
            IProgress<RegressionPipelineResult> iterationCallback = null)
        {
            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.AutoFit(trainData, validationData, label, inferredColumns,
                settings.StoppingCriteria.MaxIterations,  preprocessor, TaskKind.Regression, OptimizingMetric.RSquared);

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
            IDataView trainData, string label, IDataView validationData = null, IEstimator<ITransformer> preprocessor = null, AutoFitSettings settings = null,
            InferredColumn[] inferredColumns = null, CancellationToken cancellationToken = default(CancellationToken), 
            IProgress<BinaryClassificationPipelineResult> iterationCallback = null)
        {
            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.AutoFit(trainData, validationData, label, inferredColumns,
                settings.StoppingCriteria.MaxIterations, preprocessor, TaskKind.BinaryClassification, OptimizingMetric.Accuracy);

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
            IDataView trainData, string label, IDataView validationData = null, IEstimator<ITransformer> preprocessor = null, AutoFitSettings settings = null,
            InferredColumn[] inferredColumns = null, CancellationToken cancellationToken = default(CancellationToken), 
            IProgress<MulticlassClassificationPipelineResult> iterationCallback = null)
        {
            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.AutoFit(trainData, validationData, label, inferredColumns,
                settings.StoppingCriteria.MaxIterations, preprocessor, TaskKind.MulticlassClassification, OptimizingMetric.Accuracy);

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
        public TextLoader BuildTextLoader()
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

    public class RunHistory
    {
        
    }

    public class AutoFitSettings
    {
        public ExperimentStoppingCriteria StoppingCriteria;
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
        public int MaxIterations;
        public int TimeOutInMinutes;
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
        RegressionLightGBM,
        ClassficationRandomForest,
        ClassificationLightGBM
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
        internal readonly Pipeline Pipeline;

        public PipelineResult(ITransformer model, IDataView scoredValidationData, Pipeline pipeline = null)
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
        Hyperparams
    }

    // Following exception is used when the data
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
}
