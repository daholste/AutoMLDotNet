// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.PipelineInference2
{
    public static class AutoMlUtils
    {
        public static Random Random = new Random();

        public static void Assert(bool boolVal, string message = null)
        {
            if(!boolVal)
            {
                message = message ?? "Assertion failed";
                throw new Exception(message);
            }
        }

        /// <summary>
        /// Using the dependencyMapping and included transforms, computes which subset of columns in dataSample
        /// will be present in the final transformed dataset when only the transforms present are applied.
        /// </summary>
        private static int[] GetExcludedColumnIndices(TransformInference.SuggestedTransform[] includedTransforms, IDataView dataSample,
            AutoInference.DependencyMap dependencyMapping)
        {
            List<int> includedColumnIndices = new List<int>();

            // For every column, see if either present in initial dataset, or
            // produced by a transform used in current pipeline.
            for (int columnIndex = 0; columnIndex < dataSample.Schema.Count; columnIndex++)
            {
                // Create ColumnInfo object for indexing dictionary
                var colInfo = new AutoInference.ColumnInfo
                {
                    Name = dataSample.Schema[columnIndex].Name,
                    ItemType = dataSample.Schema[columnIndex].Type.ItemType(),
                    IsHidden = dataSample.Schema[columnIndex].IsHidden
                };

                // Exclude all hidden and non-numeric columns
                if (colInfo.IsHidden || !colInfo.ItemType.IsNumber())
                    continue;

                foreach (var level in dependencyMapping.Keys.Reverse())
                {
                    var levelResponsibilities = dependencyMapping[level];

                    if (!levelResponsibilities.ContainsKey(colInfo))
                        continue;

                    // Include any numeric column present in initial dataset. Does not need
                    // any transforms applied to be present in final dataset.
                    if (level == 0 && colInfo.ItemType.IsNumber() && levelResponsibilities[colInfo].Count == 0)
                    {
                        includedColumnIndices.Add(columnIndex);
                        break;
                    }

                    // If column could not have been produced by transforms at this level, move down to the next level.
                    if (levelResponsibilities[colInfo].Count == 0)
                        continue;

                    // Check if could have been produced by any transform in this pipeline
                    if (levelResponsibilities[colInfo].Any(t => includedTransforms.Contains(t)))
                        includedColumnIndices.Add(columnIndex);
                }
            }

            // Exclude all columns not discovered by our inclusion process
            return Enumerable.Range(0, dataSample.Schema.Count).Except(includedColumnIndices).ToArray();
        }

        public static IDataView ApplyTransformSet(IDataView data, TransformInference.SuggestedTransform[] transforms)
        {
            foreach(var transform in transforms)
            {
                data = transform.Estimator.Fit(data).Transform(data);
            }
            return data;
        }

        public static long TransformsToBitmask(TransformInference.SuggestedTransform[] transforms) =>
            transforms.Aggregate(0, (current, t) => current | 1 << t.AtomicGroupId);

        /// <summary>
        /// Gets a final transform to concatenate all numeric columns into a "Features" vector column.
        /// Note: May return empty set if Features column already present and is only relevant numeric column.
        /// (In other words, if there would be nothing for that concatenate transform to do.)
        /// </summary>
        private static TransformInference.SuggestedTransform[] GetFinalFeatureConcat(MLContext env,
            IDataView dataSample, int[] excludedColumnIndices, int level, int atomicIdOffset)
        {
            var finalArgs = new TransformInference.Arguments
            {
                EstimatedSampleFraction = 1.0,
                ExcludeFeaturesConcatTransforms = false,
                ExcludedColumnIndices = excludedColumnIndices
            };

            var featuresConcatTransforms = TransformInference.InferConcatNumericFeatures(env, dataSample, finalArgs);

            for (int i = 0; i < featuresConcatTransforms.Length; i++)
            {
                featuresConcatTransforms[i].RoutingStructure.Level = level;
                featuresConcatTransforms[i].AtomicGroupId += atomicIdOffset;
            }

            return featuresConcatTransforms.ToArray();
        }

        /// <summary>
        /// Exposed version of the method.
        /// </summary>
        public static TransformInference.SuggestedTransform[] GetFinalFeatureConcat(MLContext env, IDataView data,
            AutoInference.DependencyMap dependencyMapping, TransformInference.SuggestedTransform[] selectedTransforms,
            TransformInference.SuggestedTransform[] allTransforms)
        {
            int level = 1;
            int atomicGroupLimit = 0;
            if (allTransforms.Length != 0)
            {
                level = allTransforms.Max(t => t.RoutingStructure.Level) + 1;
                atomicGroupLimit = allTransforms.Max(t => t.AtomicGroupId) + 1;
            }
            var excludedColumnIndices = GetExcludedColumnIndices(selectedTransforms, data, dependencyMapping);
            return GetFinalFeatureConcat(env, data, excludedColumnIndices, level, atomicGroupLimit);
        }

        /// <summary>
        /// Creates a dictionary mapping column names to the transforms which could have produced them.
        /// </summary>
        public static AutoInference.LevelDependencyMap ComputeColumnResponsibilities(IDataView transformedData,
            TransformInference.SuggestedTransform[] appliedTransforms)
        {
            var mapping = new AutoInference.LevelDependencyMap();
            for (int i = 0; i < transformedData.Schema.Count; i++)
            {
                if (transformedData.Schema[i].IsHidden)
                    continue;
                var colInfo = new AutoInference.ColumnInfo
                {
                    IsHidden = false,
                    ItemType = transformedData.Schema[i].Type.ItemType(),
                    Name = transformedData.Schema[i].Name
                };
                mapping.Add(colInfo, appliedTransforms.Where(t =>
                    t.RoutingStructure.ColumnsProduced.Any(o => o.Name == colInfo.Name &&
                    o.IsNumeric == transformedData.Schema[i].Type.ItemType().IsNumber())).ToList());
            }
            return mapping;
        }

        public static IValueGenerator ToIValueGenerator(SweepableParam param)
        {
            if (param is SweepableLongParam sweepableLongParam)
            {
                var args = new LongParamArguments
                {
                    Min = sweepableLongParam.Min,
                    Max = sweepableLongParam.Max,
                    LogBase = sweepableLongParam.IsLogScale,
                    Name = sweepableLongParam.Name,
                    StepSize = sweepableLongParam.StepSize
                };
                if (sweepableLongParam.NumSteps != null)
                    args.NumSteps = (int)sweepableLongParam.NumSteps;
                return new LongValueGenerator(args);
            }

            if (param is SweepableFloatParam sweepableFloatParam)
            {
                var args = new FloatParamArguments
                {
                    Min = sweepableFloatParam.Min,
                    Max = sweepableFloatParam.Max,
                    LogBase = sweepableFloatParam.IsLogScale,
                    Name = sweepableFloatParam.Name,
                    StepSize = sweepableFloatParam.StepSize
                };
                if (sweepableFloatParam.NumSteps != null)
                    args.NumSteps = (int)sweepableFloatParam.NumSteps;
                return new FloatValueGenerator(args);
            }

            if (param is SweepableDiscreteParam sweepableDiscreteParam)
            {
                var args = new DiscreteParamArguments
                {
                    Name = sweepableDiscreteParam.Name,
                    Values = sweepableDiscreteParam.Options.Select(o => o.ToString()).ToArray()
                };
                return new DiscreteValueGenerator(args);
            }

            throw new Exception($"Sweeping only supported for Discrete, Long, and Float parameter types. Unrecognized type {param.GetType()}");
        }

        private static void SetValue(FieldInfo fi, IComparable value, object entryPointObj, Type propertyType)
        {
            if (propertyType == value?.GetType())
                fi.SetValue(entryPointObj, value);
            else if (propertyType == typeof(double) && value is float)
                fi.SetValue(entryPointObj, Convert.ToDouble(value));
            else if (propertyType == typeof(int) && value is long)
                fi.SetValue(entryPointObj, Convert.ToInt32(value));
            else if (propertyType == typeof(long) && value is int)
                fi.SetValue(entryPointObj, Convert.ToInt64(value));
        }

        /// <summary>
        /// Updates properties of entryPointObj instance based on the values in sweepParams
        /// </summary>
        public static void UpdateFields(object entryPointObj, IEnumerable<SweepableParam> sweepParams)
        {
            foreach (var param in sweepParams)
            {
                try
                {
                    // Only updates property if param.value isn't null and
                    // param has a name of property.
                    if(param.RawValue == null)
                    {
                        continue;
                    }
                    var fi = entryPointObj.GetType().GetField(param.Name);
                    var propType = Nullable.GetUnderlyingType(fi.FieldType) ?? fi.FieldType;

                    if (param is SweepableDiscreteParam dp)
                    {
                        var optIndex = (int)dp.RawValue;
                        //Contracts.Assert(0 <= optIndex && optIndex < dp.Options.Length, $"Options index out of range: {optIndex}");
                        var option = dp.Options[optIndex].ToString().ToLower();

                        // Handle <Auto> string values in sweep params
                        if (option == "auto" || option == "<auto>" || option == "< auto >")
                        {
                            //Check if nullable type, in which case 'null' is the auto value.
                            if (Nullable.GetUnderlyingType(fi.FieldType) != null)
                                fi.SetValue(entryPointObj, null);
                            else if (fi.FieldType.IsEnum)
                            {
                                // Check if there is an enum option named Auto
                                var enumDict = fi.FieldType.GetEnumValues().Cast<int>()
                                    .ToDictionary(v => Enum.GetName(fi.FieldType, v), v => v);
                                if (enumDict.ContainsKey("Auto"))
                                    fi.SetValue(entryPointObj, enumDict["Auto"]);
                            }
                        }
                        else
                            SetValue(fi, (IComparable)dp.Options[optIndex], entryPointObj, propType);
                    }
                    else
                        SetValue(fi, param.RawValue, entryPointObj, propType);
                }
                catch (Exception)
                {
                    // hack: make better error message
                    throw new Exception("cannot set learner parameter");
                }
            }
        }

        public static double ProcessWeight(double weight, double maxWeight, bool isMaximizingMetric) =>
            isMaximizingMetric ? weight : maxWeight - weight;

        public static IRunResult ConvertToRunResult(RecipeInference.SuggestedRecipe.SuggestedLearner learner, double result, bool isMetricMaximizing)
        {
            return new RunResult(learner.PipelineNode.BuildParameterSet(), result, isMetricMaximizing);
        }

        public static IRunResult[] ConvertToRunResults(PipelinePattern[] history, bool isMetricMaximizing)
        {
            return history.Select(h => ConvertToRunResult(h.Learner, h.Result, isMetricMaximizing)).ToArray();
        }

        public static IValueGenerator[] ConvertToValueGenerators(IEnumerable<SweepableParam> hps)
        {
            var results = new IValueGenerator[hps.Count()];

            for (int i = 0; i < hps.Count(); i++)
            {
                switch (hps.ElementAt(i))
                {
                    case SweepableDiscreteParam dp:
                        var dpArgs = new DiscreteParamArguments()
                        {
                            Name = dp.Name,
                            Values = dp.Options.Select(o => o.ToString()).ToArray()
                        };
                        results[i] = new DiscreteValueGenerator(dpArgs);
                        break;

                    case SweepableFloatParam fp:
                        var fpArgs = new FloatParamArguments()
                        {
                            Name = fp.Name,
                            Min = fp.Min,
                            Max = fp.Max,
                            LogBase = fp.IsLogScale,
                        };
                        if (fp.NumSteps.HasValue)
                        {
                            fpArgs.NumSteps = fp.NumSteps.Value;
                        }
                        if (fp.StepSize.HasValue)
                        {
                            fpArgs.StepSize = fp.StepSize.Value;
                        }
                        results[i] = new FloatValueGenerator(fpArgs);
                        break;

                    case SweepableLongParam lp:
                        var lpArgs = new LongParamArguments()
                        {
                            Name = lp.Name,
                            Min = lp.Min,
                            Max = lp.Max,
                            LogBase = lp.IsLogScale
                        };
                        if (lp.NumSteps.HasValue)
                        {
                            lpArgs.NumSteps = lp.NumSteps.Value;
                        }
                        if (lp.StepSize.HasValue)
                        {
                            lpArgs.StepSize = lp.StepSize.Value;
                        }
                        results[i] = new LongValueGenerator(lpArgs);
                        break;
                }
            }
            return results;
        }
    }
}
