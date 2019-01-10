// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using static Microsoft.ML.Auto.TransformInference.ColumnRoutingStructure;

namespace Microsoft.ML.Auto
{
    /// <summary>
    /// A runnable pipeline. Contains a learner and set of transforms,
    /// along with a RunSummary if it has already been exectued.
    /// </summary>
    internal class InferredPipeline
    {
        private readonly MLContext _context;
        public readonly IList<SuggestedTransform> Transforms;
        public readonly SuggestedTrainer Trainer;

        public InferredPipeline(IEnumerable<SuggestedTransform> transforms,
            SuggestedTrainer trainer,
            MLContext context = null)
        {
            Transforms = transforms.Select(t => t.Clone()).ToList();
            Trainer = trainer.Clone();
            _context = context ?? new MLContext();
            AddNormalizationTransforms();
        }
        
        public override string ToString() => $"{Trainer}+{string.Join("+", Transforms.Select(t => t.ToString()))}";

        public override bool Equals(object obj)
        {
            var pipeline = obj as InferredPipeline;
            if(pipeline == null)
            {
                return false;
            }
            return pipeline.ToString() == this.ToString();
        }

        public override int GetHashCode()
        {
            return ToString().GetHashCode();
        }

        public Pipeline ToPipeline()
        {
            var pipelineElements = new List<PipelineNode>();
            foreach(var transform in Transforms)
            {
                pipelineElements.Add(transform.ToPipelineNode());
            }
            pipelineElements.Add(Trainer.ToPipelineNode());
            return new Pipeline(pipelineElements.ToArray());
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
            var learner = Trainer.BuildTrainer(_context);

            // append learner to pipeline
            pipeline = pipeline.Append(learner);

            return pipeline.Fit(trainData);
        }

        private void AddNormalizationTransforms()
        {
            // get learner
            var learner = Trainer.BuildTrainer(_context);

            // only add normalization if learner needs it
            if (!learner.Info.NeedNormalization)
            {
                return;
            }

            var estimator = _context.Transforms.Normalize(DefaultColumnNames.Features);
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