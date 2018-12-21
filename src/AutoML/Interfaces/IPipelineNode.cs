// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Core.Data;
using Microsoft.ML.PipelineInference2;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FactorizationMachine;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.Online;

namespace Microsoft.ML.PipelineInference2
{
    public abstract class PipelineNodeBase
    {
        public virtual ParameterSet HyperSweeperParamSet { get; set; }

        protected void PropagateParamSetValues(ParameterSet hyperParams,
            IEnumerable<SweepableParam> sweepParams)
        {
            var spMap = sweepParams.ToDictionary(sp => sp.Name);

            foreach (var hp in hyperParams)
            {
                //Contracts.Assert(spMap.ContainsKey(hp.Name));
                if(spMap.ContainsKey(hp.Name))
                {
                    var sp = spMap[hp.Name];
                    sp.SetUsingValueText(hp.ValueText);
                }
            }
        }
    }

    public sealed class TransformPipelineNode
    {
        public readonly IEstimator<ITransformer> Estimator;

        public TransformPipelineNode(IEstimator<ITransformer> estimator)
        {
            Estimator = estimator;
        }

        public TransformPipelineNode Clone()
        {
            return new TransformPipelineNode(Estimator);
        }
    }

    public sealed class TrainerPipelineNode : PipelineNodeBase
    {
        public IEnumerable<SweepableParam> SweepParams { get; }

        private readonly MLContext _mlContext;
        private readonly ILearnerCatalogItem _learnerCatalogItem;

        public TrainerPipelineNode(MLContext mlContext,
            ILearnerCatalogItem learnerCatalogItem,
            IEnumerable<SweepableParam> sweepParams = null,
            ParameterSet hyperParameterSet = null)
        {
            _mlContext = mlContext;
            _learnerCatalogItem = learnerCatalogItem;
            SweepParams = sweepParams.ToArray();
            HyperSweeperParamSet = hyperParameterSet?.Clone();

            // Make sure sweep params and param set are consistent.
            if (HyperSweeperParamSet != null)
            {
                PropagateParamSetValues(HyperSweeperParamSet, SweepParams);
            }
        }

        public ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> BuildTrainer(MLContext env)
        {
            var sweepParams = HyperSweeperParamSet == null ? null : SweepParams;
            return _learnerCatalogItem.CreateInstance(_mlContext, sweepParams);
        }

        public TrainerPipelineNode Clone() => new TrainerPipelineNode(_mlContext, _learnerCatalogItem, SweepParams, HyperSweeperParamSet);

        public override string ToString()
        {
            return $"{_learnerCatalogItem.GetLearnerName()}{{{string.Join(", ", SweepParams.Where(p => p != null && p.RawValue != null).Select(p => $"{p.Name}:{p.ProcessedValue()}"))}}}";
        }

        public ParameterSet BuildParameterSet()
        {
            return BuildParameterSet(SweepParams);
        }

        private static ParameterSet BuildParameterSet(IEnumerable<SweepableParam> sweepParams)
        {
            var paramValues = new List<IParameterValue>();
            foreach (var sweepParam in sweepParams)
            {
                IParameterValue paramValue = null;
                switch (sweepParam)
                {
                    case SweepableDiscreteParam dp:
                        paramValue = new StringParameterValue(dp.Name, dp.ProcessedValue().ToString());
                        break;
                    case SweepableFloatParam fp:
                        paramValue = new FloatParameterValue(fp.Name, (float)fp.RawValue);
                        break;
                    case SweepableLongParam lp:
                        paramValue = new LongParameterValue(lp.Name, (long)lp.RawValue);
                        break;
                        //default: throw?
                }
                paramValues.Add(paramValue);
            }
            return new ParameterSet(paramValues);
        }
    }
}
