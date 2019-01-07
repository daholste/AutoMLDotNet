using Microsoft.ML;
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
            var columnInference = mlContext.Data.InferColumns(trainDataPath, label, true);
            var textLoader = columnInference.BuildTextLoader();
            var data = textLoader.Read(trainDataPath);
            var pipeline = BinaryClassificationPipelineSuggester.GetFirstPipeline(data, label);
        }
    }
}
