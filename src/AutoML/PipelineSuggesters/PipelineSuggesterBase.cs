// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Auto
{
    internal abstract class PipelineSuggesterBase : IPipelineSuggester
    {
        protected IEnumerable<SuggestedTrainer> AvailableTrainers;
        protected IEnumerable<SuggestedTransform> AvailableTransforms;

        private readonly HashSet<InferredPipeline> FailedPipelines;
        protected readonly bool IsMaximizingMetric;
        protected readonly MLContext MLContext;

        protected PipelineSuggesterBase(MLContext mlContext, bool isMaximizingMetric,
            IEnumerable<SuggestedTrainer> availableTrainers, IEnumerable<SuggestedTransform> availableTransforms)
        {
            AvailableTrainers = availableTrainers;
            AvailableTransforms = availableTransforms;

            FailedPipelines = new HashSet<InferredPipeline>();
            MLContext = mlContext;
            IsMaximizingMetric = isMaximizingMetric;
        }

        public abstract IEnumerable<InferredPipeline> GetNextPipelines(IEnumerable<PipelineRunResult> history, int numberOfCandidates);

        public void MarkPipelineAsFailed(InferredPipeline failedPipeline)
        {
            FailedPipelines.Add(failedPipeline);
        }

        protected bool HasPipelineFailed(InferredPipeline failedPipeline)
        {
            return FailedPipelines.Contains(failedPipeline);
        }
    }
}
