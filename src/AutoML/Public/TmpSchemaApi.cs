using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Auto
{
    public static class TmpSchemaApi
    {
        public static TextLoader.Arguments InferTextLoaderArguments(MLContext env,
            string dataFile, string labelColName)
        {
            var sample = TextFileSample.CreateFromFullFile(dataFile);
            var splitResult = TextFileContents.TrySplitColumns(sample, TextFileContents.DefaultSeparators);
            var columnPurposes = InferenceUtils.InferColumnPurposes(env, sample, splitResult, out var hasHeader, labelColName);
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
