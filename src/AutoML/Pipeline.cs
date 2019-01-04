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
using static Microsoft.ML.Auto.TransformInference.ColumnRoutingStructure;

namespace Microsoft.ML.Auto
{
    /// <summary>
    /// A runnable pipeline. Contains a learner and set of transforms,
    /// along with a RunSummary if it has already been exectued.
    /// </summary>
    internal class Pipeline
    {
        private readonly MLContext _mlContext;
        public readonly IList<SuggestedTransform> Transforms;
        public readonly SuggestedTrainer Trainer;

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

        public override int GetHashCode()
        {
            return ToString().GetHashCode();
        }

        public Public.Pipeline ToObjectModel()
        {
            var pipelineElements = new List<Public.PipelineElement>();
            foreach(var transform in Transforms)
            {
                pipelineElements.Add(transform.ToObjectModel());
            }
            pipelineElements.Add(Trainer.ToObjectModel());
            return new Public.Pipeline(pipelineElements.ToArray());
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