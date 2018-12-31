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
        IEnumerable<Pipeline> GetNextPipelines(IEnumerable<Pipeline> history, int numberOfCandidates);

        void UpdateTrainers(IEnumerable<SuggestedTrainer> availableLearners);

        void UpdateTransforms(IEnumerable<SuggestedTransform> availableTransforms);

        void MarkPipelineAsFailed(Pipeline failedPipeline);
    }
}
