// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    /// <summary>
    /// Automatic inference of column purposes for the data view.
    /// This is used in the context of text import wizard, but can be used outside as well.
    /// </summary>
    internal static class PurposeInference
    {
        public const int MaxRowsToRead = 1000;

        public class Column
        {
            public readonly int ColumnIndex;
            public readonly ColumnPurpose Purpose;
            public readonly DataKind ItemKind;

            public Column(int columnIndex, ColumnPurpose purpose, DataKind itemKind)
            {
                ColumnIndex = columnIndex;
                Purpose = purpose;
                ItemKind = itemKind;
            }
        }

        /// <summary>
        /// The design is the same as for <see cref="ColumnTypeInference"/>: there's a sequence of 'experts'
        /// that each look at all the columns. Every expert may or may not assign the 'answer' (suggested purpose)
        /// to a column. If the expert needs some information about the column (for example, the column values),
        /// this information is lazily calculated by the column object, not the expert itself, to allow the reuse
        /// of the same information by another expert.
        /// </summary>
        private interface IPurposeInferenceExpert
        {
            void Apply(IntermediateColumn[] columns);
        }

        private class IntermediateColumn
        {
            private readonly IDataView _data;
            private readonly int _columnId;
            private bool _isPurposeSuggested;
            private ColumnPurpose _suggestedPurpose;
            private readonly Lazy<ColumnType> _type;
            private readonly Lazy<string> _columnName;
            private object _cachedData;

            public int ColumnIndex { get { return _columnId; } }

            public bool IsPurposeSuggested { get { return _isPurposeSuggested; } }

            public ColumnPurpose SuggestedPurpose
            {
                get { return _suggestedPurpose; }
                set
                {
                    _suggestedPurpose = value;
                    _isPurposeSuggested = true;
                }
            }

            public ColumnType Type { get { return _type.Value; } }

            public string ColumnName { get { return _columnName.Value; } }

            public IntermediateColumn(IDataView data, int columnId, ColumnPurpose suggestedPurpose = ColumnPurpose.Ignore)
            {
                _data = data;
                _columnId = columnId;
                _type = new Lazy<ColumnType>(() => _data.Schema[_columnId].Type);
                _columnName = new Lazy<string>(() => _data.Schema[_columnId].Name);
                _suggestedPurpose = suggestedPurpose;
            }

            public Column GetColumn()
            {
                return new Column(_columnId, _suggestedPurpose, _type.Value.RawKind());
            }

            public T[] GetData<T>()
            {
                if (_cachedData is T[])
                    return _cachedData as T[];

                var results = new List<T>();
                using (var cursor = _data.GetRowCursor(id => id == _columnId))
                {
                    var getter = cursor.GetGetter<T>(_columnId);
                    while (cursor.MoveNext())
                    {
                        T value = default(T);
                        getter(ref value);
                        results.Add(value);
                    }
                }

                T[] resultArray;
                _cachedData = resultArray = results.ToArray();
                return resultArray;
            }
        }

        private static class Experts
        {
            internal sealed class HeaderComprehension : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested)
                            continue;
                        else if (Regex.IsMatch(column.ColumnName, @"^m_queryid$", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Group;
                        else if (Regex.IsMatch(column.ColumnName, @"group", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Group;
                        else if (Regex.IsMatch(column.ColumnName, @"^m_\w+id$", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Name;
                        else if (Regex.IsMatch(column.ColumnName, @"^id$", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Name;
                        else if (Regex.IsMatch(column.ColumnName, @"^m_", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Ignore;
                        else
                            continue;
                    }
                }
            }

            internal sealed class TextClassification : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    string[] commonImageExtensions = { ".bmp", ".dib", ".rle", ".jpg", ".jpeg", ".jpe", ".jfif", ".gif", ".tif", ".tiff", ".png" };
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested || !column.Type.IsText())
                            continue;
                        var data = column.GetData<ReadOnlyMemory<char>>();

                        long sumLength = 0;
                        int sumSpaces = 0;
                        var seen = new HashSet<string>();
                        int imagePathCount = 0;
                        foreach (var span in data)
                        {
                            sumLength += span.Length;
                            seen.Add(span.ToString());
                            string spanStr = span.ToString();
                            sumSpaces += spanStr.Count(x => x == ' ');

                            foreach (var ext in commonImageExtensions)
                            {
                                if (spanStr.EndsWith(ext, StringComparison.OrdinalIgnoreCase))
                                {
                                    imagePathCount++;
                                    break;
                                }
                            }
                        }

                        if (imagePathCount < data.Length - 1)
                        {
                            Double avgLength = 1.0 * sumLength / data.Length;
                            Double cardinalityRatio = 1.0 * seen.Count / data.Length;
                            Double avgSpaces = 1.0 * sumSpaces / data.Length;
                            if (cardinalityRatio < 0.7 || seen.Count < 100)
                                column.SuggestedPurpose = ColumnPurpose.CategoricalFeature;
                            else if (cardinalityRatio >= 0.85 && (avgLength > 30 || avgSpaces >= 1))
                                column.SuggestedPurpose = ColumnPurpose.TextFeature;
                            else if (cardinalityRatio >= 0.9)
                                column.SuggestedPurpose = ColumnPurpose.Name;
                            else
                                column.SuggestedPurpose = ColumnPurpose.Ignore;
                        }
                        else
                            column.SuggestedPurpose = ColumnPurpose.ImagePath;
                    }
                }
            }

            internal sealed class NumericAreFeatures : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested)
                            continue;
                        if (column.Type.ItemType().IsNumber())
                            column.SuggestedPurpose = ColumnPurpose.NumericFeature;
                    }
                }
            }

            internal sealed class BooleanProcessing : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested)
                            continue;
                        if (column.Type.ItemType().IsBool())
                            column.SuggestedPurpose = ColumnPurpose.NumericFeature;
                    }
                }
            }

            internal sealed class TextArraysAreText : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested)
                            continue;
                        if (column.Type.IsVector() && column.Type.ItemType().IsText())
                            column.SuggestedPurpose = ColumnPurpose.TextFeature;
                    }
                }
            }

            internal sealed class IgnoreEverythingElse : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (!column.IsPurposeSuggested)
                            column.SuggestedPurpose = ColumnPurpose.Ignore;
                    }
                }
            }
        }

        private static IEnumerable<IPurposeInferenceExpert> GetExperts()
        {
            // Each of the experts respects the decisions of all the experts above.

            // Use column names to suggest purpose.
            yield return new Experts.HeaderComprehension();
            // Single-value text columns may be category, name, text or ignore.
            yield return new Experts.TextClassification();
            // Vector-value text columns are always treated as text.
            // REVIEW: could be improved.
            yield return new Experts.TextArraysAreText();
            // Check column on boolean only values.
            yield return new Experts.BooleanProcessing();
            // All numeric columns are features.
            yield return new Experts.NumericAreFeatures();
            // Everything else is ignored.
            yield return new Experts.IgnoreEverythingElse();
        }

        /// <summary>
        /// Auto-detect purpose for the data view columns.
        /// </summary>
        public static PurposeInference.Column[] InferPurposes(MLContext context, IDataView data, string label,
            PurposeInference.Column[] columnOverrides = null)
        {
            var labelColumn = data.GetColumn(label);

            // select columns to include in inferencing
            var columnIndices = CalcIncludedIndices(data.Schema.Count, labelColumn.Index, columnOverrides);

            // do purpose inferencing
            var intermediateCols = InferPurposes(context, data, columnIndices);

            // result to return to caller
            var result = new PurposeInference.Column[data.Schema.Count];

            // add label column to result
            result[labelColumn.Index] = (new IntermediateColumn(data, labelColumn.Index, ColumnPurpose.Label)).GetColumn();

            // add inferred columns to result
            foreach (var intermediateCol in intermediateCols)
            {
                result[intermediateCol.ColumnIndex] = intermediateCol.GetColumn();
            }

            // add overrides to result
            if(columnOverrides != null)
            {
                foreach (var columnOverride in columnOverrides)
                {
                    result[columnOverride.ColumnIndex] = columnOverride;
                }
            }

            return result;
        }
        
        private static IntermediateColumn[] InferPurposes(MLContext context, IDataView data, IEnumerable<int> columnIndices)
        {
            data = data.Take(MaxRowsToRead);
            var cols = columnIndices.Select(x => new IntermediateColumn(data, x)).ToArray();
            foreach (var expert in GetExperts())
            {
                expert.Apply(cols);
            }
            return cols;
        }

        private static IEnumerable<int> CalcIncludedIndices(int columnCount,
            int labelIndex, 
            PurposeInference.Column[] columnOverrides = null)
        {
            var allIndices = new HashSet<int>(Enumerable.Range(0, columnCount));
            allIndices.Remove(labelIndex);
            if(columnOverrides != null)
            {
                foreach (var columnOverride in columnOverrides)
                {
                    allIndices.Remove(columnOverride.ColumnIndex);
                }
            }
            return allIndices;
        }
    }
}
