using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Auto;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Auto.Public;

namespace Samples
{
    public class MulticlassClassification
    {
        public static void Run()
        {
            const string trainDataPath = @"C:\data\train.csv";
            const string validationDataPath = @"C:\data\valid.csv";
            const string testDataPath = @"C:\data\test.csv";

            var mlContext = new MLContext();

            // auto-load data from disk
            var trainData = mlContext.Data.AutoRead(trainDataPath);
            var validationData = mlContext.Data.AutoRead(validationDataPath);
            var testData = mlContext.Data.AutoRead(testDataPath);

            // run AutoML & train model
            var autoMlResult = mlContext.MulticlassClassification.AutoFit(trainData, "Label", validationData,
                settings: new AutoFitSettings() { MaxIterations = 14 });
            // get best AutoML model
            var model = autoMlResult.BestPipeline.Model;

            // run AutoML on test data
            var transformedOutput = model.Transform(testData);
            var results = mlContext.BinaryClassification.Evaluate(transformedOutput);
            Console.WriteLine($"Model Accuracy: {results.Accuracy}\r\n");

            Console.ReadLine();
        }
    }
}
