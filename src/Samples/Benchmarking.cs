using System;
using Microsoft.ML;
using Microsoft.ML.Auto;

namespace Samples
{
    public static class Benchmarking
    {
        const string TrainDataPath = @"D:\SplitDatasets\EverGreen_train.csv";
        const string ValidationDataPath = @"D:\SplitDatasets\EverGreen_valid.csv";
        const string TestDataPath = @"D:\SplitDatasets\EverGreen_test.csv";
        const string Label = "Label";

        public static void Run()
        {
            var context = new MLContext();
            var columnInference = context.Data.InferColumns(TrainDataPath, Label, true);
            var textLoader = context.Data.CreateTextReader(columnInference);
            var trainData = textLoader.Read(TrainDataPath);
            var validationData = textLoader.Read(ValidationDataPath);
            var testData = textLoader.Read(TestDataPath);
            var best = context.BinaryClassification.AutoFit(trainData, Label, validationData, settings:
                new AutoFitSettings()
                {
                    StoppingCriteria = new ExperimentStoppingCriteria()
                    {
                        MaxIterations = 200,
                        TimeOutInMinutes = 1000000000
                    }
                });
            var scoredTestData = best.BestPipeline.Model.Transform(testData);
            var testDataMetrics = context.BinaryClassification.EvaluateNonCalibrated(scoredTestData);

            Console.WriteLine(testDataMetrics.Accuracy);
            Console.ReadLine();
        }
    }
}
