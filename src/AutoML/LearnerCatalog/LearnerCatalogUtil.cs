using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    public enum LearnerNames
    {
        AveragedPerceptron,
    }

    public static class LearnerCatalogUtil
    {
        public static readonly IEnumerable<SweepableParam> AveragedLinearArgsSweepableParams =
            new SweepableParam[]
            {
                new SweepableDiscreteParam("LearningRate", new object[] { 0.01, 0.1, 0.5, 1.0 }),
                new SweepableDiscreteParam("DecreaseLearningRate", new object[] { false, true }),
                new SweepableFloatParam("L2RegularizerWeight", 0.0f, 0.4f),
            };

        public static readonly IEnumerable<SweepableParam> OnlineLinearArgsSweepableParams =
            new SweepableParam[]
            {
                new SweepableLongParam("NumIterations", 1, 100, stepSize: 10, isLogScale: true),
                new SweepableFloatParam("InitWtsDiameter", 0.0f, 1.0f, numSteps: 5),
                new SweepableDiscreteParam("Shuffle", new object[] { false, true }),
            };

        public static readonly IEnumerable<SweepableParam> TreeArgsSweepableParams =
           new SweepableParam[]
           {
                new SweepableLongParam("NumLeaves", 2, 128, isLogScale: true, stepSize: 4),
                new SweepableDiscreteParam("MinDocumentsInLeafs", new object[] { 1, 10, 50 }),
                new SweepableDiscreteParam("NumTrees", new object[] { 20, 100, 500 }),
                new SweepableFloatParam("LearningRates", 0.025f, 0.4f, isLogScale: true),
                new SweepableFloatParam("Shrinkage", 0.025f, 4f, isLogScale: true),
           };

        public static Action<T> CreateArgsFunc<T>(IEnumerable<SweepableParam> sweepParams)
        {
            Action<T> argsFunc = null;
            if (sweepParams != null)
            {
                argsFunc = (args) => AutoMlUtils.UpdatePropertiesAndFields(args, sweepParams);
            }
            return argsFunc;
        }
    }
}
