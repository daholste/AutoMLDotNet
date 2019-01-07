﻿using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Auto
{
    public class Pipeline
    {
        public readonly PipelineNode[] Elements;

        public Pipeline(PipelineNode[] elements)
        {
            Elements = elements;
        }
    }

    public class PipelineNode
    {
        public readonly string Name;
        public readonly PipelineNodeType ElementType;
        public readonly string[] InColumns;
        public readonly string[] OutColumns;
        public readonly IDictionary<string, object> Properties;

        public PipelineNode(string name, PipelineNodeType elementType,
            string[] inColumns, string[] outColumns,
            IDictionary<string, object> properties)
        {
            Name = name;
            ElementType = elementType;
            InColumns = inColumns;
            OutColumns = outColumns;
            Properties = properties;
        }
    }

    public enum PipelineNodeType
    {
        Transform,
        Trainer
    }
}
