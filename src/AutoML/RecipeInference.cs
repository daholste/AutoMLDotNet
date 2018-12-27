// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.Online;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.PipelineInference2;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Learners;

namespace Microsoft.ML.PipelineInference2
{
    public static class RecipeInference
    {
        public readonly struct SuggestedRecipe
        {
            public readonly string Description;
            public readonly TransformInference.SuggestedTransform[] Transforms;
            public struct SuggestedLearner
            {
                public string Settings;
                public TrainerPipelineNode PipelineNode;
                public string LearnerName;

                public SuggestedLearner Clone()
                {
                    return new SuggestedLearner
                    {
                        Settings = Settings,
                        PipelineNode = PipelineNode.Clone(),
                        LearnerName = LearnerName
                    };
                }

                public override string ToString() => PipelineNode.ToString();
            }

            public readonly SuggestedLearner[] Learners;
        }

        public static TextLoader.Arguments MyAutoMlInferTextLoaderArguments(MLContext env,
            string dataFile, string labelColName)
        {
            //var h = env.Register("InferRecipesFromData", seed: 0, verbose: false);
            //using (var ch = h.Start("InferRecipesFromData"))
            //{
            var sample = TextFileSample.CreateFromFullFile(dataFile);
            var splitResult = TextFileContents.TrySplitColumns(sample, TextFileContents.DefaultSeparators);
            var columnPurposes = InferenceUtils.InferColumnPurposes(env, sample, splitResult,
                out var hasHeader, labelColName);
            return new TextLoader.Arguments
            {
                Column = ColumnGroupingInference.GenerateLoaderColumns(columnPurposes),
                HasHeader = true,
                Separator = splitResult.Separator,
                AllowSparse = splitResult.AllowSparse,
                AllowQuoting = splitResult.AllowQuote
            };
            //}
        }

        /// <summary>
        /// Given a predictor type returns a set of all permissible learners (with their sweeper params, if defined).
        /// </summary>
        /// <returns>Array of viable learners.</returns>
        public static SuggestedRecipe.SuggestedLearner[] AllowedLearners(MLContext mlContext, MacroUtils.TrainerKinds task,
            int maxNumIterations)
        {
            var learnerCatalogItems = LearnerCatalog.GetLearners(task, maxNumIterations);

            var learners = new List<SuggestedRecipe.SuggestedLearner>();
            foreach (var learnerCatalogItem in learnerCatalogItems)
            {
                var sweepParams = learnerCatalogItem.GetHyperparamSweepRanges();
                var learnerName = learnerCatalogItem.GetLearnerName();
                var learner = new SuggestedRecipe.SuggestedLearner
                {
                    PipelineNode = new TrainerPipelineNode(mlContext, learnerCatalogItem, sweepParams),
                    LearnerName = learnerName.ToString()
                };
                learners.Add(learner);
            }
            return learners.ToArray();
        }
    }
}
