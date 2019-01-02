using Microsoft.ML.Runtime.EntryPoints;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    public static class MyGlobals
    {
        public static string OutputDir = ".";
        public static ISet<string> FailedPipelineHashes = new HashSet<string>();
    }
}
