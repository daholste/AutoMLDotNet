// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;

namespace Microsoft.ML.Auto
{
    internal class PipelineRunResult
    {
        public readonly Pipeline Pipeline;
        public readonly double Result;
        public readonly ITransformer Model;

        public PipelineRunResult(Pipeline pipeline, double result, ITransformer model)
        {
            Pipeline = pipeline;
            Result = result;
            Model = model;
        }
    }
}
