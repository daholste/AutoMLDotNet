// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.PipelineInference2
{
    /// <summary>
    /// AutoML engine that goes through all lerners using defaults. Need to decide
    /// how to handle which transforms to try.
    /// </summary>
    public sealed class DefaultsEngine : PipelineOptimizerBase
    {
        private int _currentLearnerIndex;

        public DefaultsEngine(MLContext env) : base(env)
        {
            _currentLearnerIndex = 0;
        }

        public override PipelinePattern[] GetNextCandidates(IEnumerable<PipelinePattern> history, int numCandidates)
        {
            var candidates = new List<PipelinePattern>();

            while (candidates.Count < numCandidates)
            {
                //Contracts.Assert(0 <= _currentLearnerIndex && _currentLearnerIndex < AvailableLearners.Length);

                // Select hyperparameters and transforms based on learner and history.
                var learner = AvailableLearners[_currentLearnerIndex];
                var pipeline = new PipelinePattern(SampleTransforms(out var transformsBitMask),
                    learner, "", Env);

                // Keep only valid pipelines.
                candidates.Add(pipeline);

                // Update current index
                _currentLearnerIndex = (_currentLearnerIndex + 1) % AvailableLearners.Length;
            }

            return candidates.ToArray();
        }

        private TransformInference.SuggestedTransform[] SampleTransforms(out long transformsBitMask)
        {
            // For now, return all transforms.
            var sampledTransforms = AvailableTransforms.ToList();
            transformsBitMask = AutoMlUtils.TransformsToBitmask(sampledTransforms.ToArray());

            //sampledTransforms = sampledTransforms.Take(sampledTransforms.Count() - 3).ToList();

            // Add final features concat transform.
            sampledTransforms.AddRange(AutoMlUtils.GetFinalFeatureConcat(Env, FullyTransformedData,
                DependencyMapping, sampledTransforms.ToArray(), AvailableTransforms));

            return sampledTransforms.ToArray();
        }
    }
}
