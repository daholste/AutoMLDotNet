using Microsoft.ML;
using Microsoft.ML.Auto;

namespace Samples
{
    public static class GetFirstPipeline
    {
        const string trainDataPath = @"C:\data\sample_train2.csv";
        const string label = "Label";

        public static void Run()
        {
            var mlContext = new MLContext();
            var columnInference = mlContext.Data.InferColumns(trainDataPath, label, true);
            var textLoader = mlContext.Data.CreateTextReader(columnInference);
            var data = textLoader.Read(trainDataPath);
            var pipeline = mlContext.BinaryClassification.GetPipeline(data, label);
        }
    }
}
