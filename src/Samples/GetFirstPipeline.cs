using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Auto.Public;

namespace Samples
{
    public static class GetFirstPipeline
    {
        const string trainDataPath = @"C:\data\sample_train2.csv";
        const string label = "Label";

        public static void Run()
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.AutoRead(trainDataPath, label);
            var pipeline = BinaryClassificationPipelineSuggester.GetFirstPipeline(data, label);
        }
    }
}
