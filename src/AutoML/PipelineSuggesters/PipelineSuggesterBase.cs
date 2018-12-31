// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.PipelineInference2
{
    internal abstract class PipelineSuggesterBase : IPipelineSuggester
    {
        protected IEnumerable<SuggestedTransform> AvailableTransforms;
        protected IEnumerable<SuggestedTrainer> AvailableTrainers;
        protected readonly MLContext MLContext;
        protected readonly HashSet<string> VisitedPipelines;        
        protected bool IsMaximizingMetric;

        private readonly HashSet<string> FailedPipelines;

        protected PipelineSuggesterBase(MLContext mlContext, bool isMaximizingMetric)
        {
            MLContext = mlContext;
            IsMaximizingMetric = isMaximizingMetric;
            VisitedPipelines = new HashSet<string>();
            FailedPipelines = new HashSet<string>();
        }

        public abstract IEnumerable<Pipeline> GetNextPipelines(IEnumerable<Pipeline> history, int numberOfCandidates);

        public virtual void UpdateTrainers(IEnumerable<SuggestedTrainer> availableTrainers)
        {
            AvailableTrainers = availableTrainers;
        }

        public virtual void UpdateTransforms(IEnumerable<SuggestedTransform> availableTransforms)
        {
            AvailableTransforms = availableTransforms;
        }

        public void MarkPipelineAsFailed(Pipeline failedPipeline)
        {
            FailedPipelines.Add(failedPipeline.ToString());
        }

        public bool HasPipelineFailed(Pipeline failedPipeline)
        {
            return FailedPipelines.Contains(failedPipeline.ToString());
        }
    }
}
