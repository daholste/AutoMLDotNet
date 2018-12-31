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
}
