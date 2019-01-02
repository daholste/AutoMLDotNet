using Microsoft.ML.Core.Data;
using Microsoft.ML.PipelineInference2;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace AutoML.Public
{
    public static class PipelineSuggester
    {
        public static Pipeline[] GetNextPipeLines(RunHistory history)
        {
            throw new NotImplementedException();
        }
    }

    public static class TransformExtensions
    {
        public static IEstimator<ITransformer> InferTransforms(this TransformsCatalog catalog, IDataView dataView)
        {
            throw new NotImplementedException();
        }
    }

    public static class DataExtensions
    {
        // Delimiter, header, column datatype inference
        public static (bool hasHeader, string separator, TextLoader.Column[] columns) InferColumns(this DataOperations catalog, string path, bool? hasHeader = null, string separator = null, TextLoader.Column[] columns = null)
        {
            throw new NotImplementedException();
        }

        // Auto reader (includes column inference)
        public static IDataView AutoRead(this DataOperations catalog, Stream stream)
        {
            throw new NotImplementedException();
        }

        public static IDataView AutoRead(this DataOperations catalog, string path)
        {
            throw new NotImplementedException();
        }

        public static IDataView AutoRead(this DataOperations catalog, IMultiStreamSource source)
        {
            throw new NotImplementedException();
        }

        // Purpose Inference
        public static (ColumnPurpose purpose, IEnumerable<TextLoader.Column> columns)[] InferColumPurposes(this DataOperations catalog, IDataView dataView, (ColumnPurpose purpose, IEnumerable<TextLoader.Column> columns)[] knownColumnPurposes = null)
        {
            throw new NotImplementedException();
        }

        // Task inference
        public static MachineLearningTaskType InferTask(this DataOperations catalog, IDataView dataView)
        {
            throw new NotImplementedException();
        }



        public enum MachineLearningTaskType
        {
            Regression,
            BinaryClassification,
            MultiClassClassification
        }
    }

    public class RunHistory
    {

    }
}
