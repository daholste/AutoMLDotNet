// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.PipelineInference2
{
    public sealed class IterationTerminator : ITerminator
    {
        private readonly int _finalHistoryLength;
        private readonly Stopwatch _stopwatch;
        private readonly TimeSpan? _maxRunTime;

        public sealed class Arguments
        {
            //[Argument(ArgumentType.Required, HelpText = "Total number of iterations.", ShortName = "length")]
            public int FinalHistoryLength;

            public ITerminator CreateComponent(MLContext env) => new IterationTerminator(FinalHistoryLength);
        }

        public IterationTerminator(int finalHistoryLength, TimeSpan? maxRunTime = null,
            Stopwatch stopwatch = null)
        {
            _finalHistoryLength = finalHistoryLength;
            _stopwatch = stopwatch;
            _maxRunTime = maxRunTime;
        }

        public bool ShouldTerminate(IEnumerable<PipelinePattern> history)
        {
            if(_maxRunTime != null && _stopwatch.Elapsed > _maxRunTime)
            {
                return true;
            }
            return history.ToArray().Length >= _finalHistoryLength;
        }

        public int RemainingIterations(IEnumerable<PipelinePattern> history) =>
            _finalHistoryLength - history.ToArray().Length;
    }
}
