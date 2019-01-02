// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Auto
{
    internal class VBufferUtils
    {
        public static bool HasNaNs(in VBuffer<Single> buffer)
        {
            var values = buffer.GetValues();
            for (int i = 0; i < values.Length; i++)
            {
                if (Single.IsNaN(values[i]))
                    return true;
            }
            return false;
        }
    }
}
