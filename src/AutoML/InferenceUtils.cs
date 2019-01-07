// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Auto
{
    internal static class InferenceUtils
    {
        public static IDataView Take(this IDataView data, int count)
        {
            // REVIEW: This should take an env as a parameter, not create one.
            var env = new MLContext();
            var take = SkipTakeFilter.Create(env, new SkipTakeFilter.TakeArguments { Count = count }, data);
            return CacheCore(take, env);
        }

        private static IDataView CacheCore(IDataView data, MLContext env)
        {
            return new CacheDataView(env, data, Enumerable.Range(0, data.Schema.Count).ToArray());
        }
    }

    public enum ColumnPurpose
    {
        Ignore = 0,
        Name = 1,
        Label = 2,
        NumericFeature = 3,
        CategoricalFeature = 4,
        TextFeature = 5,
        Weight = 6,
        Group = 7,
        ImagePath = 8
    }
}
