// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Auto;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Conversions;
using Microsoft.ML.Transforms.Text;
using static Microsoft.ML.Auto.TransformInference;

namespace Microsoft.ML.Auto
{
    internal class SuggestedTransform
    {
        public readonly IEstimator<ITransformer> Estimator;
        public readonly IDictionary<string, string> Properties;
        // Stores which columns are consumed by this transform,
        // and which are produced, at which level.
        public ColumnRoutingStructure RoutingStructure { get; set; }

        public SuggestedTransform(IEstimator<ITransformer> estimator,
            ColumnRoutingStructure routingStructure = null, IDictionary<string, string> properties = null)
        {
            Estimator = estimator;
            RoutingStructure = routingStructure;
            Properties = properties;
        }

        public SuggestedTransform Clone()
        {
            return new SuggestedTransform(Estimator, RoutingStructure, Properties);
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append(Estimator.GetType().FullName);
            sb.Append("{");
            if (RoutingStructure.ColumnsProduced.Count() > 1)
            {
                for (var i = 0; i < RoutingStructure.ColumnsProduced.Count(); i++)
                {
                    sb.Append($" col={RoutingStructure.ColumnsProduced[i].Name}:{RoutingStructure.ColumnsConsumed[i].Name}");
                }
            }
            else
            {
                sb.Append($" col={RoutingStructure.ColumnsProduced.First().Name}:{string.Join(",", RoutingStructure.ColumnsConsumed.Select(c => c.Name))}");
            }
            if (Properties != null)
            {
                foreach (var property in Properties)
                {
                    sb.Append($" {property.Key}={property.Value}");
                }
            }
            sb.Append("}");
            return sb.ToString();
        }

        public Auto.ObjectModel.PipelineElement ToObjectModel()
        {
            var elementProperties = new Dictionary<string, object>();
            elementProperties["InputColumns"] = RoutingStructure.ColumnsConsumed.Select(c => c.Name);
            elementProperties["OutputColumns"] = RoutingStructure.ColumnsProduced.Select(c => c.Name);
            if (Properties != null)
            {
                foreach (var property in Properties)
                {
                    elementProperties[property.Key] = property.Value;
                }
            }
            return new Auto.ObjectModel.PipelineElement(Estimator.GetType().Name, Auto.ObjectModel.PipelineElementType.Transform, elementProperties);
        }
    }

    /// <summary>
    /// Auto-generate set of transforms for the data view, given the purposes of specified columns.
    ///
    /// The design is the same as for <see cref="ColumnTypeInference"/>: there's a sequence of 'experts'
    /// that each look at all the columns. Every expert may or may not suggest additional transforms.
    /// If the expert needs some information about the column (for example, the column values),
    /// this information is lazily calculated by the column object, not the expert itself, to allow the reuse
    /// of the same information by another expert.
    /// </summary>
    internal static class TransformInference
    {
        public sealed class Arguments
        {
            /// <summary>
            /// Relative size of the inspected data view vs. the 'real' data size.
            /// </summary>
            public Double EstimatedSampleFraction;
            public bool ExcludeFeaturesConcatTransforms;
            public int[] ExcludedColumnIndices;

            public Arguments()
            {
                EstimatedSampleFraction = 1;
                ExcludeFeaturesConcatTransforms = false;
                ExcludedColumnIndices = new int[] { };
            }
        }

        private const int MaxRowsToRead = 1000;

        public readonly struct InferenceResult
        {
            public readonly SuggestedTransform[] SuggestedTransforms;

            public InferenceResult(SuggestedTransform[] suggestedTransforms)
            {
                SuggestedTransforms = suggestedTransforms;
            }
        }

        internal class IntermediateColumn
        {
            private readonly IDataView _data;
            private readonly int _columnId;
            private readonly ColumnPurpose _purpose;
            private readonly Lazy<ColumnType> _type;
            private readonly Lazy<string> _columnName;
            private readonly Lazy<bool> _hasMissing;

            public int ColumnId { get { return _columnId; } }
            public ColumnPurpose Purpose { get { return _purpose; } }
            public ColumnType Type { get { return _type.Value; } }
            public string ColumnName { get { return _columnName.Value; } }
            public bool HasMissing { get { return _hasMissing.Value; } }

            public IntermediateColumn(IDataView data, PurposeInference.Column column)
            {
                _data = data;
                _columnId = column.ColumnIndex;
                _purpose = column.Purpose;
                _type = new Lazy<ColumnType>(() => _data.Schema[_columnId].Type);
                _columnName = new Lazy<string>(() => _data.Schema[_columnId].Name);
                _hasMissing = new Lazy<bool>(() =>
                {
                    if (Type.ItemType() != NumberType.R4)
                        return false;
                    return Type.IsVector() ? HasMissingVector() : HasMissingOne();
                });
            }

            public string GetTempColumnName(string tag = null) => _data.Schema.GetTemporaryColumnName(tag);

            private bool HasMissingOne()
            {
                using (var cursor = _data.GetRowCursor(x => x == _columnId))
                {
                    var getter = cursor.GetGetter<Single>(_columnId);
                    var value = default(Single);
                    while (cursor.MoveNext())
                    {
                        getter(ref value);
                        if (Single.IsNaN(value))
                            return true;
                    }
                    return false;
                }
            }

            private bool HasMissingVector()
            {
                using (var cursor = _data.GetRowCursor(x => x == _columnId))
                {
                    var getter = cursor.GetGetter<VBuffer<Single>>(_columnId);
                    var value = default(VBuffer<Single>);
                    while (cursor.MoveNext())
                    {
                        getter(ref value);
                        if (VBufferUtils.HasNaNs(value))
                            return true;
                    }
                    return false;
                }
            }

            public void GetUniqueValueCounts<T>(out int uniqueValueCount, out int singletonCount, out int rowCount)
            {
                var seen = new HashSet<string>();
                var singletons = new HashSet<string>();
                rowCount = 0;
                using (var cursor = _data.GetRowCursor(x => x == _columnId))
                {
                    var getter = cursor.GetGetter<T>(_columnId);
                    while (cursor.MoveNext())
                    {
                        var value = default(T);
                        getter(ref value);
                        var s = value.ToString();
                        if (seen.Add(s))
                            singletons.Add(s);
                        else
                            singletons.Remove(s);
                        rowCount++;
                    }
                    uniqueValueCount = seen.Count;
                    singletonCount = singletons.Count;
                }
            }
        }

        public sealed class ColumnRoutingStructure : IEquatable<ColumnRoutingStructure>
        {
            public struct AnnotatedName
            {
                public string Name { get; set; }
                public bool IsNumeric { get; set; }

                public bool Equals(AnnotatedName an)
                {
                    return an.Name == Name &&
                           an.IsNumeric == IsNumeric;
                }

                public override string ToString() => $"{Name}({IsNumeric})";
            }

            public AnnotatedName[] ColumnsConsumed { get; }
            public AnnotatedName[] ColumnsProduced { get; }

            public ColumnRoutingStructure(AnnotatedName[] columnsConsumed, AnnotatedName[] columnsProduced)
            {
                ColumnsConsumed = columnsConsumed;
                ColumnsProduced = columnsProduced;
            }

            public bool Equals(ColumnRoutingStructure obj)
            {
                return obj != null &&
                       obj.ColumnsConsumed.All(cc => ColumnsConsumed.Any(cc.Equals)) &&
                       obj.ColumnsProduced.All(cp => ColumnsProduced.Any(cp.Equals));
            }
        }

        public interface ITransformInferenceExpert
        {
            bool IncludeFeaturesOverride { get; set; }

            IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs);
        }

        public abstract class TransformInferenceExpertBase : ITransformInferenceExpert
        {
            public bool IncludeFeaturesOverride { get; set; }

            public abstract IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs);

            protected readonly MLContext Env;

            public TransformInferenceExpertBase()
            {
                Env = new MLContext();
            }
        }

        private static IEnumerable<ITransformInferenceExpert> GetExperts()
        {
            // The expert work independently of each other, the sequence is irrelevant
            // (it only determines the sequence of resulting transforms).

            // For text labels, convert to categories.
            yield return new Experts.AutoLabel();

            // For group ID column, rename to GroupId and hash, if text.
            // REVIEW: this is only sufficient if we discard the possibility of hash collisions, and don't care
            // about the group Id cardinality (we don't for ranking).
            yield return new Experts.GroupIdHashRename();

            // For name column, rename to Name (or, if multiple and text, concat and rename to Name).
            yield return new Experts.NameColumnConcatRename();

            // Check cardinality and type for numeric labels.
            yield return new Experts.LabelAdvisory();

            // For boolean columns use convert transform
            yield return new Experts.Boolean();

            // For categorical columns, use Cat transform.
            yield return new Experts.Categorical();

            // For text columns, use TextTransform.
            yield return new Experts.Text();

            // If numeric column has missing values, use Missing transform.
            yield return new Experts.NumericMissing();

            // If there's more than one feature column, concat all into Features. If it isn't called 'Features', rename it.
            yield return new Experts.FeaturesColumnConcatRenameNumericOnly();

            // For text columns, also use TextTransform with Unigram + trichar.
            yield return new Experts.TextUniGramTriGram();
        }

        public static class Experts
        {
            public sealed class AutoLabel : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs)
                {
                    var lastLabelColId = Array.FindLastIndex(columns, x => x.Purpose == ColumnPurpose.Label);
                    if (lastLabelColId < 0)
                        yield break;

                    var col = columns[lastLabelColId];
                    
                    var columnName = new StringBuilder();
                    columnName.Append(col.ColumnName);

                    if (col.Type.IsText())
                    {
                        col.GetUniqueValueCounts<ReadOnlyMemory<char>>(out var unique, out var _, out var _);
                        //ch.Info("Label column '{0}' is text. Suggested auto-labeling.", col.ColumnName);

                        string dest = DefaultColumnNames.Label;
                        string source = columnName.ToString();
                        var input = new ValueToKeyMappingEstimator(Env, source, dest);

                        var routingStructure = new ColumnRoutingStructure(
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = false, Name = source}
                            },
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = true, Name = dest}
                            }
                        );
                        yield return new SuggestedTransform(input, routingStructure);

                        if (unique == 1)
                        {
                            //ch.Warning("Label column '{0}' has only one value in the sample. Maybe the label column is incorrect?", col.ColumnName);
                        }
                        else if (unique > 100)
                        {
                            //ch.Warning("Label column '{0}' has {1} different values in the sample. Multi-class classification might not be desirable with so many values.", col.ColumnName, unique);
                        }
                        else if (unique > 2)
                        {
                            //ch.Info("Label column '{0}' has {1} different values in the sample.", col.ColumnName, unique);
                        }
                    }
                    else if (col.ColumnName != DefaultColumnNames.Label)
                    {
                        string dest = DefaultColumnNames.Label;
                        string source = columnName.ToString();
                        var input = new ColumnCopyingEstimator(Env, source, dest);

                        var routingStructure = new ColumnRoutingStructure(
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = true, Name = source}
                            },
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = true, Name = dest}
                            }
                        );

                        yield return new SuggestedTransform(input, routingStructure);
                    }
                }
            }

            public sealed class GroupIdHashRename : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs)
                {
                    var firstGroupColId = Array.FindIndex(columns, x => x.Purpose == ColumnPurpose.Group);
                    if (firstGroupColId < 0)
                        yield break;

                    var col = columns[firstGroupColId];

                    var columnName = new StringBuilder();
                    columnName.AppendFormat("{0}", col.ColumnName);

                    if (col.Type.IsText())
                    {
                        //ch.Info("Group Id column '{0}' is text. Suggested hashing.", col.ColumnName);
                        // REVIEW: we could potentially apply HashJoin to vectors of text.
                        string dest = DefaultColumnNames.GroupId;
                        string source = columnName.ToString();
                        var input = new OneHotHashEncodingEstimator(Env, new OneHotHashEncodingEstimator.ColumnInfo(dest, source));

                        var routingStructure = new ColumnRoutingStructure(
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = false, Name = source}
                            },
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = true, Name = dest}
                            }
                        );

                        string[] outputColNames = new string[] { DefaultColumnNames.GroupId };
                        yield return new SuggestedTransform(input, routingStructure);
                    }
                    else if (col.ColumnName != DefaultColumnNames.GroupId)
                    {
                        //ch.Warning("Group Id column '{0}' is not text. Couldn't determine correct transformation.");
                        string dest = DefaultColumnNames.GroupId;
                        string source = columnName.ToString();
                        var input = new ColumnCopyingEstimator(Env, source, dest);

                        var routingStructure = new ColumnRoutingStructure(
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = true, Name = source}
                            },
                            new[]
                            {
                                new ColumnRoutingStructure.AnnotatedName {IsNumeric = true, Name = dest}
                            }
                        );

                        yield return new SuggestedTransform(input, routingStructure);
                    }
                }
            }

            public sealed class LabelAdvisory : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs)
                {
                    var firstLabelColId = Array.FindIndex(columns, x => x.Purpose == ColumnPurpose.Label);
                    if (firstLabelColId < 0)
                    {
                        //ch.Info("Label column not present in the dataset. This is likely an unsupervised learning problem.");
                        yield break;
                    }

                    var col = columns[firstLabelColId];
                    if (col.Type.IsText())
                        yield break;

                    if (col.Type.IsKnownSizeVector() && col.Type.ItemType() == NumberType.R4)
                    {
                        //ch.Info("Label column '{0}' has type {1}, this can be only a multi-output regression problem.", col.ColumnName, col.Type);
                        yield break;
                    }

                    if (col.Type != NumberType.R4)
                    {
                        //ch.Warning("Label column '{0}' has type {1} which is not supported by any learner.", col.ColumnName, col.Type);
                        yield break;
                    }

                    int unique;
                    int singleton;
                    int total;
                    col.GetUniqueValueCounts<Single>(out unique, out singleton, out total);

                    if (unique == 1)
                    {
                        //ch.Warning("Label column '{0}' has only one value in the sample. Maybe the label column is incorrect?", col.ColumnName);
                    }
                    else if (unique > 100)
                    {
                        //ch.Info("Label column '{0}' has {1} different values in the sample. This is likely a regression problem.", col.ColumnName, unique);
                    }
                    else if (unique > 2)
                    {
                        //ch.Info("Label column '{0}' has {1} different values in the sample. This can be treated as multi-class or regression problem.", col.ColumnName, unique);
                    }
                }
            }

            public sealed class Categorical : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs)
                {
                    bool foundCat = false;
                    bool foundCatHash = false;
                    var catColumnsNew = new List<OneHotEncodingEstimator.ColumnInfo>();
                    var catHashColumnsNew = new List<OneHotHashEncodingEstimator.ColumnInfo>();
                    var featureCols = new List<string>();

                    foreach (var column in columns)
                    {
                        if (!column.Type.ItemType().IsText() || column.Purpose != ColumnPurpose.CategoricalFeature)
                            continue;

                        var columnName = new StringBuilder();
                        columnName.AppendFormat("{0}", column.ColumnName);

                        if (IsDictionaryOk(column, inferenceArgs.EstimatedSampleFraction))
                        {
                            foundCat = true;
                            catColumnsNew.Add(new OneHotEncodingEstimator.ColumnInfo(columnName.ToString(), columnName.ToString()));
                        }
                        else
                        {
                            //ch.Info("Categorical column '{0}' has extremely high cardinality. Suggested hash-based category encoding.", column.ColumnName);
                            foundCatHash = true;
                            catHashColumnsNew.Add(new OneHotHashEncodingEstimator.ColumnInfo(columnName.ToString(), columnName.ToString()));
                        }
                    }

                    if (foundCat)
                    {
                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            catColumnsNew.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = c.Output }).ToArray();
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            catColumnsNew.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = c.Output }).ToArray();
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);

                        var input = new OneHotEncodingEstimator(Env, catColumnsNew.ToArray());
                        featureCols.AddRange(catColumnsNew.Select(c => c.Output));

                        //ch.Info("Suggested dictionary-based category encoding for categorical columns.");
                        yield return new SuggestedTransform(input, routingStructure);
                    }

                    if (foundCatHash)
                    {
                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            catHashColumnsNew.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = c.HashInfo.Output }).ToArray();
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            catHashColumnsNew.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = c.HashInfo.Output }).ToArray();
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);

                        var input = new OneHotHashEncodingEstimator(Env, catHashColumnsNew.ToArray());

                        //ch.Info("Suggested hash-based category encoding for categorical columns.");
                        yield return new SuggestedTransform(input, routingStructure);
                    }

                    if (!inferenceArgs.ExcludeFeaturesConcatTransforms && featureCols.Count > 0)
                    {
                        yield return InferenceHelpers.GetRemainingFeatures(featureCols, columns, GetType(), IncludeFeaturesOverride);
                        IncludeFeaturesOverride = true;
                    }
                }

                private bool IsDictionaryOk(IntermediateColumn column, Double dataSampleFraction)
                {
                    if (column.Type.IsVector())
                        return false;
                    //Contracts.Assert(dataSampleFraction > 0);
                    int total;
                    int unique;
                    int singletons;
                    // REVIEW: replace with proper Good-Turing estimation.
                    // REVIEW: This looks correct; cf. equation (8) of Katz S. "Estimation of Probabilities from
                    // Sparse Data for the Language Model Component of a Speech Recognizer" (1987), taking into account that
                    // the singleton count was estimated from a fraction of the data (and assuming the estimate is
                    // roughly the same for the entire sample).
                    column.GetUniqueValueCounts<ReadOnlyMemory<char>>(out unique, out singletons, out total);
                    var expectedUnseenValues = singletons / dataSampleFraction;
                    return expectedUnseenValues < 1000 && unique < 10000;
                }
            }

            public sealed class Boolean : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs)
                {
                    var columnName = new StringBuilder();
                    var newColumns = new List<TypeConvertingTransformer.ColumnInfo>();

                    foreach (var column in columns)
                    {
                        if (!column.Type.ItemType().IsBool() || column.Purpose != ColumnPurpose.NumericFeature)
                            continue;
                        columnName.AppendFormat("{0}", column.ColumnName);

                        newColumns.Add(new TypeConvertingTransformer.ColumnInfo(columnName.ToString(),
                            columnName.ToString(), DataKind.R4));
                    }

                    if (columnName.Length > 0)
                    {
                        //ch.Info("Suggested conversion to numeric for boolean features.");
                        var input = new TypeConvertingEstimator(Env, newColumns.ToArray());
                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            newColumns.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = c.Input }).ToArray();
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            newColumns.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = c.Output }).ToArray();
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                        yield return new SuggestedTransform(input, routingStructure);

                        // Concat featurized columns into existing feature column, if transformed at least one column.
                        if (!inferenceArgs.ExcludeFeaturesConcatTransforms)
                        {
                            yield return InferenceHelpers.GetRemainingFeatures(newColumns.Select(c => c.Output).ToList(),
                                columns, GetType(), IncludeFeaturesOverride);
                            IncludeFeaturesOverride = true;
                        }
                    }
                }
            }

            internal static class InferenceHelpers
            {
                public static SuggestedTransform GetRemainingFeatures(List<string> newCols, IntermediateColumn[] existingColumns,
                    Type currentType, bool includeFeaturesOverride)
                {
                    // Pick up existing features columns, if they exist
                    var featuresColumnsCount = existingColumns.Count(col =>
                     (col.Purpose == ColumnPurpose.NumericFeature) &&
                     (col.ColumnName == DefaultColumnNames.Features));
                    if (includeFeaturesOverride || featuresColumnsCount > 0)
                        newCols.Insert(0, DefaultColumnNames.Features);
                    return InferenceHelpers.ConcatColumnsIntoOne(newCols, DefaultColumnNames.Features, currentType, true);
                }

                public static SuggestedTransform ConcatColumnsIntoOne(List<string> columnNames, string concatColumnName,
                    Type transformType, bool isNumeric)
                {
                    StringBuilder columnName = new StringBuilder();

                    columnNames.ForEach(column =>
                    {
                        columnName.AppendFormat("{0}", column);
                    });

                    string columnsToConcat = string.Join(",", columnNames);

                    var env = new MLContext();
                    var input = new ColumnConcatenatingEstimator(env, concatColumnName, columnNames.ToArray());

                    // Not sure if resulting columns will be numeric or text, since concat can apply to either.
                    ColumnRoutingStructure.AnnotatedName[] columnsSource =
                        columnNames.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = isNumeric, Name = c }).ToArray();
                    ColumnRoutingStructure.AnnotatedName[] columnsDest =
                        new[] { new ColumnRoutingStructure.AnnotatedName { IsNumeric = isNumeric, Name = concatColumnName } };
                    var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);

                    return new SuggestedTransform(input, routingStructure);
                }

                public static SuggestedTransform TextTransformUnigramTriChar(MLContext env, string srcColumn, string dstColumn)
                {
                    var input = new TextFeaturizingEstimator(env, srcColumn, dstColumn)
                    {
                        //WordFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 1 },
                        //CharFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 3 }
                    };

                    return TextTransform(srcColumn, dstColumn, input);
                }

                public static SuggestedTransform TextTransformBigramTriChar(MLContext env, string srcColumn, string dstColumn, Type transformType)
                {
                    var input = new TextFeaturizingEstimator(env, srcColumn, dstColumn)
                    {
                        //WordFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 2 },
                        //CharFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 3 }
                    };

                    return TextTransform(srcColumn, dstColumn, input);
                }

                public static SuggestedTransform TextTransform(string srcColumn, string dstColumn, IEstimator<ITransformer> estimator)
                {
                    ColumnRoutingStructure.AnnotatedName[] columnsSource =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = srcColumn } };
                    ColumnRoutingStructure.AnnotatedName[] columnsDest =
                        { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = dstColumn } };
                    var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                    return new SuggestedTransform(estimator, routingStructure);
                }
            }

            public sealed class Text : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs)
                {
                    var featureCols = new List<string>();

                    foreach (var column in columns)
                    {
                        if (!column.Type.ItemType().IsText() || column.Purpose != ColumnPurpose.TextFeature)
                            continue;
                        //ch.Info("Suggested text featurization for text column '{0}'.", column.ColumnName);

                        var columnDestSuffix = "_tf";
                        var columnNameSafe = column.ColumnName;

                        string columnDestRenamed = $"{columnNameSafe}{columnDestSuffix}";

                        featureCols.Add(columnDestRenamed);
                        var input = new TextFeaturizingEstimator(Env, columnDestRenamed, columnNameSafe);
                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = columnNameSafe} };
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = columnDestRenamed} };
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                        yield return new SuggestedTransform(input, routingStructure);
                    }

                    // Concat text featurized columns into existing feature column, if transformed at least one column.
                    if (!inferenceArgs.ExcludeFeaturesConcatTransforms && featureCols.Count > 0)
                    {
                        yield return InferenceHelpers.GetRemainingFeatures(featureCols, columns, GetType(), IncludeFeaturesOverride);
                        IncludeFeaturesOverride = true;
                    }
                }
            }

            public sealed class TextUniGramTriGram : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs)
                {
                    List<string> textColumnNames =
                        columns.Where(
                            column => column.Type.ItemType().IsText() && column.Purpose == ColumnPurpose.TextFeature)
                            .Select(column => column.ColumnName).ToList();

                    if ((textColumnNames.Count == 0) ||
                        (columns.Count(col => col.Purpose == ColumnPurpose.Label) != 1))
                        yield break;

                    //Concat text columns into one.
                    string concatTextColumnName;
                    if (textColumnNames.Count > 1)
                    {
                        concatTextColumnName = columns[0].GetTempColumnName("TextConcat");
                        yield return
                            InferenceHelpers.ConcatColumnsIntoOne(textColumnNames, concatTextColumnName, GetType(), false);
                    }
                    else
                        concatTextColumnName = textColumnNames.First();

                    //Get Unigram + Trichar for text transform on the concatenated text column.
                    string featureTextColumn = columns[0].GetTempColumnName("FeaturesText");
                    yield return InferenceHelpers.TextTransformUnigramTriChar(Env, concatTextColumnName, featureTextColumn);

                    //Concat text featurized column into feature column.
                    List<string> featureCols = new List<string>(new[] { featureTextColumn });
                    if (columns.Any(
                            col =>
                                (col.Purpose == ColumnPurpose.NumericFeature) ||
                                (col.Purpose == ColumnPurpose.CategoricalFeature)))
                        featureCols.Add(DefaultColumnNames.Features);

                    if (!inferenceArgs.ExcludeFeaturesConcatTransforms)
                        yield return InferenceHelpers.ConcatColumnsIntoOne(featureCols, DefaultColumnNames.Features, GetType(), true);
                }
            }

            public sealed class NumericMissing : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs)
                {
                    bool found = false;
                    var columnName = new StringBuilder();
                    foreach (var column in columns)
                    {
                        if (column.Type.ItemType() != NumberType.R4 || column.Purpose != ColumnPurpose.NumericFeature)
                            continue;
                        if (!column.HasMissing)
                            continue;

                        //ch.Info("Column '{0}' has missing values. Suggested missing indicator encoding.", column.ColumnName);
                        found = true;
                        
                        columnName.AppendFormat("{0}", column.ColumnName);
                    }
                    if (found)
                    {
                        string name = columnName.ToString();
                        var input = new MissingValueIndicatorEstimator(Env, name, name);

                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = name} };
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = true, Name = name} };
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                        yield return new SuggestedTransform(input, routingStructure);
                    }
                }
            }

            internal class FeaturesColumnConcatRename : TransformInferenceExpertBase
            {
                public virtual bool IgnoreColumn(ColumnPurpose purpose)
                {
                    if (purpose != ColumnPurpose.TextFeature
                        && purpose != ColumnPurpose.CategoricalFeature
                        && purpose != ColumnPurpose.NumericFeature)
                        return true;
                    return false;
                }

                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs)
                {
                    var selectedColumns = columns.Where(c => !IgnoreColumn(c.Purpose)).ToArray();
                    var colList = selectedColumns.Select(c => c.ColumnName).ToArray();
                    bool allColumnsNumeric = selectedColumns.All(c => c.Purpose == ColumnPurpose.NumericFeature && c.Type.ItemType() != BoolType.Instance);
                    bool allColumnsNonNumeric = selectedColumns.All(c => c.Purpose != ColumnPurpose.NumericFeature);

                    if (colList.Length > 0)
                    {
                        // Check if column is named features and already numeric
                        if (colList.Length == 1 && colList[0] == DefaultColumnNames.Features && allColumnsNumeric)
                            yield break;

                        if (!allColumnsNumeric && !allColumnsNonNumeric)
                            yield break;
                        
                        List<string> columnList = new List<string>();

                        foreach (var column in colList)
                        {
                            var columnName = new StringBuilder();
                            columnName.AppendFormat("{0}", column);
                            columnList.Add(columnName.ToString());
                        }
                        
                        var input = new ColumnConcatenatingEstimator(Env, DefaultColumnNames.Features, columnList.ToArray());

                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            columnList.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = allColumnsNumeric, Name = c }).ToArray();
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = allColumnsNumeric, Name = DefaultColumnNames.Features} };
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                        yield return new SuggestedTransform(input, routingStructure);
                    }
                }
            }

            public sealed class FeaturesColumnConcatRenameIgnoreText : FeaturesColumnConcatRename, ITransformInferenceExpert
            {
                public override bool IgnoreColumn(ColumnPurpose purpose)
                {
                    return (purpose != ColumnPurpose.CategoricalFeature && purpose != ColumnPurpose.NumericFeature);
                }
            }

            public sealed class FeaturesColumnConcatRenameNumericOnly : FeaturesColumnConcatRename, ITransformInferenceExpert
            {
                public override bool IgnoreColumn(ColumnPurpose purpose)
                {
                    return (purpose != ColumnPurpose.NumericFeature);
                }
            }

            public sealed class NameColumnConcatRename : TransformInferenceExpertBase
            {
                public override IEnumerable<SuggestedTransform> Apply(IntermediateColumn[] columns, Arguments inferenceArgs)
                {
                    int count = 0;
                    bool isAllText = true;
                    var colSpec = new StringBuilder();
                    var colSpecTextOnly = new List<string>();
                    var columnList = new List<string>();

                    foreach (var column in columns)
                    {
                        var columnName = new StringBuilder();
                        if (column.Purpose != ColumnPurpose.Name)
                            continue;
                        count++;

                        if (colSpec.Length > 0)
                            colSpec.Append(",");
                        colSpec.Append(column.ColumnName);
                        
                        columnName.Append(column.ColumnName);
                        columnList.Add(columnName.ToString());

                        if (column.Type.ItemType().IsText())
                            colSpecTextOnly.Add(column.ColumnName);
                        isAllText = isAllText && column.Type.ItemType().IsText();
                    }

                    if (count == 1 && colSpec.ToString() != DefaultColumnNames.Name)
                    {
                        var columnName = new StringBuilder();
                        columnName.AppendFormat("{0}", colSpec);
                        var input = new ColumnCopyingEstimator(Env, columnName.ToString(), DefaultColumnNames.Name);
                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = columnName.ToString()} };
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = DefaultColumnNames.Name} };
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                        yield return new SuggestedTransform(input, routingStructure);
                    }
                    else if (count > 1)
                    {
                        //if (!isAllText)
                        //    ch.Warning("Not all name columns are textual. Ignored non-textual name columns.");
                        if (string.IsNullOrWhiteSpace(colSpecTextOnly.ToString()))
                            yield break;

                        //ch.Info("Suggested grouping name columns into one vector.");

                        var input = new ColumnConcatenatingEstimator(Env, DefaultColumnNames.Name, columnList.ToArray());

                        ColumnRoutingStructure.AnnotatedName[] columnsSource =
                            columnList.Select(c => new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = c }).ToArray();
                        ColumnRoutingStructure.AnnotatedName[] columnsDest =
                            { new ColumnRoutingStructure.AnnotatedName { IsNumeric = false, Name = DefaultColumnNames.Name} };
                        var routingStructure = new ColumnRoutingStructure(columnsSource, columnsDest);
                        yield return new SuggestedTransform(input, routingStructure);
                    }
                }
            }
        }

        /// <summary>
        /// Automatically infer transforms for the data view
        /// </summary>
        public static InferenceResult InferTransforms(MLContext env, IDataView data, PurposeInference.Column[] purposes, Arguments args)
        {
            data = data.Take(MaxRowsToRead);
            var cols = purposes.Where(x => !data.Schema[x.ColumnIndex].IsHidden).Select(x => new IntermediateColumn(data, x)).ToArray();
            var list = new List<SuggestedTransform>();
            var includeFeaturesOverride = false;
            foreach (var expert in GetExperts())
            {
                expert.IncludeFeaturesOverride = includeFeaturesOverride;
                SuggestedTransform[] suggestions = expert.Apply(cols, args).ToArray();
                includeFeaturesOverride |= expert.IncludeFeaturesOverride;

                list.AddRange(suggestions);
            }

            return new InferenceResult(list.ToArray());
        }

        public static SuggestedTransform[] InferTransforms(MLContext env, IDataView data, Arguments args)
        {
            //Contracts.CheckValue(env, nameof(env));
            //var h = env.Register("InferTransforms");
            //h.CheckValue(data, nameof(data));
            //h.CheckValue(args, nameof(args));
            //h.Check(args.EstimatedSampleFraction > 0);

            var dataSample = data.Take(MaxRowsToRead);

            // Infer column purposes from data sample.
            var piArgs = new PurposeInference.Arguments { MaxRowsToRead = MaxRowsToRead };
            var columnIndices = Enumerable.Range(0, dataSample.Schema.Count);
            var piResult = PurposeInference.InferPurposes(env, dataSample, columnIndices, piArgs);
            var purposes = piResult.Columns;

            // Infer transforms
            var inferenceResult = InferTransforms(env, data, purposes, args);
            return inferenceResult.SuggestedTransforms;

            // Keep viable transforms, where all transforms in the atomic group have pipeline nodes.
            // First clause: ensure transform t has a pipeline node (is runnable).
            // Second clause: ensure transform is in module catalog entry point list for this platform.
            // Third clause: ensure all members of t's atoimic group are also available, since it is all
            // or nothing for atomic groups (hence the name atomic -- 'uncuttable').
            /*return inferenceResult.SuggestedTransforms.Where(t => t.PipelineNode != null
                && availableTransforms.Any(a => a.Name.Equals(t.PipelineNode.GetEpName()))
                && !inferenceResult.SuggestedTransforms
                .Where(t2 => t2.PipelineNode == null)
                .Select(t2 => t2.AtomicGroupId)
                .Contains(t.AtomicGroupId)).ToArray();*/
        }

        public static SuggestedTransform[] InferConcatNumericFeatures(MLContext env, IDataView data, Arguments args)
        {
            //Contracts.CheckValue(env, nameof(env));
            //var h = env.Register("InferConcatNumericFeatures");
            //h.CheckValue(data, nameof(data));
            //h.CheckValue(args, nameof(args));
            //h.Check(args.EstimatedSampleFraction > 0);

            data = data.Take(MaxRowsToRead);

            // Infer column purposes from data sample.
            var piArgs = new PurposeInference.Arguments { MaxRowsToRead = MaxRowsToRead };
            var columnIndices = Enumerable.Range(0, data.Schema.Count);
            var piResult = PurposeInference.InferPurposes(env, data, columnIndices, piArgs);
            var purposes = piResult.Columns;

            var cols = purposes.Where(x => !data.Schema[x.ColumnIndex].IsHidden
                && !args.ExcludedColumnIndices.Contains(x.ColumnIndex))
                .Select(x => new IntermediateColumn(data, x))
                .ToArray();
            //using (var rootCh = h.Start("InferConcatNumericFeatures"))
            //{
            var list = new List<SuggestedTransform>();
            int atomicGroupId = 0;
            var expert = new Experts.FeaturesColumnConcatRenameNumericOnly();

            //using (var ch = h.Start(expert.GetType().ToString()))
            //{
            SuggestedTransform[] suggestions = expert.Apply(cols, args).ToArray();
            list.AddRange(suggestions);
            return list.ToArray();
        }
    }
}
