// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.PipelineInference2
{
    internal abstract class PipelineSuggesterBase : IPipelineSuggester
    {
        protected IEnumerable<SuggestedTrainer> AvailableTrainers;
        protected IEnumerable<SuggestedTransform> AvailableTransforms;

        private readonly HashSet<Pipeline> FailedPipelines;
        protected readonly bool IsMaximizingMetric;
        protected readonly MLContext MLContext;

        protected PipelineSuggesterBase(MLContext mlContext, bool isMaximizingMetric,
            IEnumerable<SuggestedTrainer> availableTrainers, IEnumerable<SuggestedTransform> availableTransforms)
        {
            AvailableTrainers = availableTrainers;
            AvailableTransforms = availableTransforms;

            FailedPipelines = new HashSet<Pipeline>();
            MLContext = mlContext;
            IsMaximizingMetric = isMaximizingMetric;
        }

        public abstract IEnumerable<Pipeline> GetNextPipelines(IEnumerable<PipelineRunResult> history, int numberOfCandidates);

        public void MarkPipelineAsFailed(Pipeline failedPipeline)
        {
            FailedPipelines.Add(failedPipeline);
        }

        protected bool HasPipelineFailed(Pipeline failedPipeline)
        {
            return FailedPipelines.Contains(failedPipeline);
        }
    }
}
