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
            // todo: respect passed-in label

            var mlContext = new MLContext();
            var availableTransforms = TransformInferenceApi.InferTransforms(mlContext, dataView);
            var availableTrainers = RecipeInference.AllowedTrainers(mlContext, TaskKind.BinaryClassification, 1);
            var pipeline = new Auto.Pipeline(availableTransforms, availableTrainers.First(), mlContext);
            return pipeline.ToObjectModel();
        }
    }

    public static class RegressionExtensions
    {
        public static RegressionResult AutoFit(this RegressionContext context,
            IDataView trainData, string label, IDataView validationData = null, IEstimator<ITransformer> preprocessor = null, AutoFitSettings settings = null,
            CancellationToken cancellationToken = default(CancellationToken), IProgress<RegressionPipelineResult> iterationCallback = null)
        {
            // todo: respect passed-in label

            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.AutoFit(trainData, validationData, settings.MaxIterations, preprocessor, TaskKind.Regression, OptimizingMetric.RSquared);

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
            CancellationToken cancellationToken = default(CancellationToken), IProgress<BinaryClassificationPipelineResult> iterationCallback = null)
        {
            // todo: respect passed-in label

            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.AutoFit(trainData, validationData, settings.MaxIterations, preprocessor, TaskKind.BinaryClassification, OptimizingMetric.Accuracy);

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
            CancellationToken cancellationToken = default(CancellationToken), IProgress<MulticlassClassificationPipelineResult> iterationCallback = null)
        {
            // todo: respect passed-in label

            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.AutoFit(trainData, validationData, settings.MaxIterations, preprocessor, TaskKind.MulticlassClassification, OptimizingMetric.Accuracy);

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
        public static IEstimator<ITransformer> InferTransforms(this TransformsCatalog catalog, IDataView data)
        {
            var mlContext = new MLContext();
            var suggestedTransforms = TransformInferenceApi.InferTransforms(mlContext, data);
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
        public static (bool hasHeader, string separator, TextLoader.Column[] columns, ColumnPurpose[] purposes) InferColumns(this DataOperations catalog, string path, bool? hasHeader = null, string separator = null, TextLoader.Column[] columns = null)
        {
            throw new NotImplementedException();
        }

        // Auto reader (includes column inference)
        public static IDataView AutoRead(this DataOperations catalog, Stream stream)
        {
            throw new NotImplementedException();
        }

        public static IDataView AutoRead(this DataOperations catalog, string path)
        {
            var mlContext = new MLContext();
            var textLoaderArgs = SchemaInferenceApi.InferTextLoaderArguments(mlContext, path, "Label"); // todo: remove label from here?
            var textLoader = new TextLoader(mlContext, textLoaderArgs);
            return textLoader.Read(path);
        }

        public static IDataView AutoRead(this DataOperations catalog, IMultiStreamSource source)
        {
            throw new NotImplementedException();
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
        internal TraceLevel TraceLevel; // Should this be controlled through code or appconfig?
        
        //remove
        internal int MaxIterations;
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

        public PipelineResult(ITransformer model, IDataView scoredValidationData)
        {
            Model = model;
            ScoredValidationData = scoredValidationData;
        }
    }
}
