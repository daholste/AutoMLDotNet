// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.PipelineInference2
{
    /// <summary>
    /// Interface that defines what an AutoML engine looks like
    /// </summary>
    internal interface IPipelineSuggester
    {
        IEnumerable<PipelinePattern> GetNextPipelines(IEnumerable<PipelinePattern> history, int numberOfCandidates);

        void UpdateLearners(IEnumerable<SuggestedLearner> availableLearners);

        void UpdateTransforms(IEnumerable<TransformInference.SuggestedTransform> availableTransforms);

        void MarkPipelineAsFailed(PipelinePattern failedPipeline);
    }

    internal abstract class PipelineSuggesterBase : IPipelineSuggester
    {
        protected IEnumerable<TransformInference.SuggestedTransform> AvailableTransforms;
        protected IEnumerable<SuggestedLearner> AvailableLearners;
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

        public abstract IEnumerable<PipelinePattern> GetNextPipelines(IEnumerable<PipelinePattern> history, int numberOfCandidates);

        public virtual void UpdateLearners(IEnumerable<SuggestedLearner> availableLearners)
        {
            AvailableLearners = availableLearners;
        }

        public virtual void UpdateTransforms(IEnumerable<TransformInference.SuggestedTransform> availableTransforms)
        {
            AvailableTransforms = availableTransforms;
        }

        public void MarkPipelineAsFailed(PipelinePattern failedPipeline)
        {
            FailedPipelines.Add(failedPipeline.ToString());
        }

        public bool HasPipelineFailed(PipelinePattern failedPipeline)
        {
            return FailedPipelines.Contains(failedPipeline.ToString());
        }
    }
}
