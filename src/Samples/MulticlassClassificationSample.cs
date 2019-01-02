using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Auto;
using Microsoft.ML.Runtime.Data;

namespace Samples
{
    public class MulticlassClassificationSample
    {
        public static void Run()
        {
            const string trainDataPath = @"C:\data\train.csv";
            const string validationDataPath = @"C:\data\valid.csv";
            const string testDataPath = @"C:\data\test.csv";

            var mlContext = new MLContext();

            // auto-infer text loader args
            var textLoaderArgs = RecipeInference.MyAutoMlInferTextLoaderArguments(mlContext, trainDataPath, "Label");

            // load data from disk
            var textLoader = new TextLoader(mlContext, textLoaderArgs);
            var trainData = textLoader.Read(trainDataPath);
            var validationData = textLoader.Read(validationDataPath);
            var testData = textLoader.Read(testDataPath);

            // run AutoML & train model
            var autoMlResult = mlContext.MulticlassClassification.AutoFit(trainData, validationData, 5);
            // get best AutoML model
            var model = autoMlResult.BestModel;
            // print all AutoML pipelines
            var allPipelines = autoMlResult.AllPipelines;
            foreach (var pipeline in allPipelines)
            {
                Console.WriteLine(pipeline);
            }

            // run AutoML on test data
            var transformedOutput = model.Transform(testData);
            var results = mlContext.BinaryClassification.Evaluate(transformedOutput);
            Console.WriteLine($"Model Accuracy: {results.Accuracy}\r\n");

            Console.ReadLine();
        }
    }
}
