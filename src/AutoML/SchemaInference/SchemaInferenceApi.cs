using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Auto
{
    internal static class SchemaInferenceApi
    {
        public static TextLoader.Arguments InferTextLoaderArguments(MLContext env,
            string dataFile, string label)
        {
            var sample = TextFileSample.CreateFromFullFile(dataFile);
            var splitResult = TextFileContents.TrySplitColumns(sample, TextFileContents.DefaultSeparators);
            var columnPurposes = InferenceUtils.InferColumnPurposes(env, sample, splitResult, out var hasHeader, label);
            return new TextLoader.Arguments
            {
                Column = ColumnGroupingInference.GenerateLoaderColumns(columnPurposes),
                HasHeader = true,
                Separator = splitResult.Separator,
                AllowSparse = splitResult.AllowSparse,
                AllowQuoting = splitResult.AllowQuote
            };
        }
    }
}
