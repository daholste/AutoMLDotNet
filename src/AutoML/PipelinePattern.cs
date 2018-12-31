// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Auto;
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
    public sealed class Pipeline
    {
        private readonly MLContext _mlContext;
        public readonly IList<SuggestedTransform> Transforms;
        public readonly SuggestedTrainer Trainer;
        public double Result { get; set; }

        public Pipeline(IEnumerable<SuggestedTransform> transforms,
            SuggestedTrainer trainer,
            MLContext mlContext)
        {
            Transforms = transforms.Select(t => t.Clone()).ToList();
            Trainer = trainer.Clone();
            _mlContext = mlContext;
            AddNormalizationTransforms();
        }
        
        public override string ToString() => $"{Trainer}+{string.Join("+", Transforms.Select(t => t.ToString()))}";

        public Auto.ObjectModel.Pipeline ToObjectModel()
        {
            var pipelineElements = new List<Auto.ObjectModel.PipelineElement>();
            foreach(var transform in Transforms)
            {
                pipelineElements.Add(transform.ToObjectModel());
            }
            pipelineElements.Add(Trainer.ToObjectModel());
            return new Auto.ObjectModel.Pipeline(pipelineElements.ToArray());
        }

        /// <summary>
        /// Runs a train-test experiment on the current pipeline
        /// </summary>
        public void RunTrainTestExperiment(IDataView trainData, IDataView validationData,
            MacroUtils.TrainerKinds task, MLContext mlContext,
            out double testMetricValue)
        {
            var pipelineTransformer = TrainTransformer(trainData);
            var scoredTestData = pipelineTransformer.Transform(validationData);
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
            var learner = Trainer.BuildTrainer(_mlContext);

            // append learner to pipeline
            pipeline = pipeline.Append(learner);

            return pipeline.Fit(trainData);
        }

        private void AddNormalizationTransforms()
        {
            // get learner
            var learner = Trainer.BuildTrainer(_mlContext);

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
            var transform = new SuggestedTransform(estimator, 
                routingStructure: routingStructure, properties: properties);
            Transforms.Add(transform);
        }
    }
}