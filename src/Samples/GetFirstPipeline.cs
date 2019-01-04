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

        public static void Run()
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.AutoRead(trainDataPath);
            var pipeline = RegressionPipelineSuggester.GetFirstPipeline(data, "Label");
        }
    }
}
