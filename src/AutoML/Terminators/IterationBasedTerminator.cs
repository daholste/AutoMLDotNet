// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.PipelineInference2
{
    public sealed class IterationBasedTerminator : ITerminator
    {
        private readonly int _finalHistoryLength;

        public IterationBasedTerminator(int finalHistoryLength)
        {
            _finalHistoryLength = finalHistoryLength;
        }

        public bool ShouldTerminate(IEnumerable<PipelinePattern> history)
        {
            return history.ToArray().Length >= _finalHistoryLength;
        }

        public int RemainingIterations(IEnumerable<PipelinePattern> history) =>
            _finalHistoryLength - history.ToArray().Length;
    }
}
