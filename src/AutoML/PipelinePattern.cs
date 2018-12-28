// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.PipelineInference2;
using Microsoft.ML.Transforms.Normalizers;
using static Microsoft.ML.PipelineInference2.TransformInference.ColumnRoutingStructure;

namespace Microsoft.ML.PipelineInference2
{
    /// <summary>
    /// A runnable pipeline. Contains a learner and set of transforms,
    /// along with a RunSummary if it has already been exectued.
    /// </summary>
    public sealed class PipelinePattern
    {
        private readonly MLContext _mlContext;
        public readonly IList<TransformInference.SuggestedTransform> Transforms;
        public readonly RecipeInference.SuggestedRecipe.SuggestedLearner Learner;
        public double Result { get; set; }

        public PipelinePattern(TransformInference.SuggestedTransform[] transforms,
            RecipeInference.SuggestedRecipe.SuggestedLearner learner,
            string loaderSettings, MLContext mlContext)
        {
            // Make sure internal pipeline nodes and sweep params are cloned, not shared.
            // Cloning the transforms and learner rather than assigning outright
            // ensures that this will be the case. Doing this here allows us to not
            // worry about changing hyperparameter values in candidate pipelines
            // possibly overwritting other pipelines.
            Transforms = transforms.Select(t => t.Clone()).ToList();
            Learner = learner.Clone();
            _mlContext = mlContext;
            AddNormalizationTransforms();
        }

        /// <summary>
        /// This method will return some indentifying string for the pipeline,
        /// based on transforms, learner, and (eventually) hyperparameters.
        /// </summary>
        public override string ToString() => $"{Learner}+{string.Join("+", Transforms.Select(t => t.ToString()))}";

        /// <summary>
        /// Runs a train-test experiment on the current pipeline
        /// </summary>
        public void RunTrainTestExperiment(IDataView trainData, IDataView testData,
            MacroUtils.TrainerKinds task, MLContext mlContext,
            out double testMetricValue)
        {
            var pipelineTransformer = TrainTransformer(trainData);
            var scoredTestData = pipelineTransformer.Transform(testData);
            testMetricValue = GetTestMetricValue(mlContext, task, scoredTestData);
        }

        private static double GetTestMetricValue(MLContext mlContext, MacroUtils.TrainerKinds task, IDataView scoredData)
        {
            if (task == MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer)
            {
                return mlContext.BinaryClassification.EvaluateNonCalibrated(scoredData).Accuracy;
            }
            if (task == MacroUtils.TrainerKinds.SignatureMultiClassClassifierTrainer)
            {
                return mlContext.MulticlassClassification.Evaluate(scoredData).AccuracyMicro;
            }
            if (task == MacroUtils.TrainerKinds.SignatureMultiClassClassifierTrainer)
            {
                return mlContext.Regression.Evaluate(scoredData).RSquared;
            }
            else
            {
                // todo: better error handling?
                throw new Exception("unsupported task");
            }
        }

        public ITransformer TrainTransformer(IDataView trainData)
        {
            IEstimator<ITransformer> pipeline = new EstimatorChain<ITransformer>();

            // append each transformer to the pipeline
            foreach (var transform in Transforms)
            {
                if(transform.Estimator != null)
                {
                    pipeline = pipeline.Append(transform.Estimator);
                }
            }

            // get learner
            var learner = Learner.PipelineNode.BuildTrainer(_mlContext);

            // append learner to pipeline
            pipeline = pipeline.Append(learner);

            return pipeline.Fit(trainData);
        }

        private void AddNormalizationTransforms()
        {
            // get learner
            var learner = Learner.PipelineNode.BuildTrainer(_mlContext);

            // only add normalization if learner needs it
            if (!learner.Info.NeedNormalization)
            {
                return;
            }

            var estimator = _mlContext.Transforms.Normalize(DefaultColumnNames.Features);
            var annotatedColNames = new[] { new AnnotatedName() { Name = DefaultColumnNames.Features, IsNumeric = true } };
            var routingStructure = new TransformInference.ColumnRoutingStructure(annotatedColNames, annotatedColNames);
            var properties = new Dictionary<string, string>()
            {
                { "mode", "MinMax" }
            };
            var transform = new TransformInference.SuggestedTransform(estimator, 
                routingStructure: routingStructure, properties: properties);
            Transforms.Add(transform);
        }
    }
}