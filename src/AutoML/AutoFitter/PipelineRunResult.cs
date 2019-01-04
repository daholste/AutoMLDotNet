// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Auto
{
    internal class PipelineRunResult
    {
        public readonly object EvaluatedMetrics;
        public readonly Pipeline Pipeline;
        public readonly double Score;
        public readonly IDataView ScoredValidationData;

        public ITransformer Model { get; set; }

        public PipelineRunResult(object evaluatedMetrics, ITransformer model, Pipeline pipeline, double score, IDataView scoredValidationData)
        {
            EvaluatedMetrics = evaluatedMetrics;
            Model = model;
            Pipeline = pipeline;
            Score = score;
            ScoredValidationData = scoredValidationData;
        }
    }
}
