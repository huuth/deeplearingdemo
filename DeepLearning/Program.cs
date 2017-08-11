using Accord.Neuro;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro.Learning;
using Accord.Neuro.Networks;
using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AForge.Neuro.Learning;
using System.IO;

namespace DeepLearning
{
    class Program
    {
        static void Main(string[] args)
        {

         
            test();
            Console.WriteLine();
     
            Console.Write("Press any key to quit ..");
            Console.ReadKey();
        }

        public static void test()
        {
            //double[][] inputs;
            //double[][] outputs;
            //double[][] testInputs;
            //double[][] testOutputs;

            //// Load ascii digits dataset.
            //inputs = DataManager.Load(@"../../../data/data.txt", out outputs);

            //// The first 500 data rows will be for training. The rest will be for testing.
            //testInputs = inputs.Skip(500).ToArray();
            //testOutputs = outputs.Skip(500).ToArray();
            //inputs = inputs.Take(500).ToArray();
            //outputs = outputs.Take(500).ToArray();
            //double[][] inputs = new double[4][] {
            //    new double[] {0, 0}, new double[] {0, 1},
            //    new double[] {1, 0}, new double[] {1, 1}
            //};
            //double[][] outputs = new double[4][] {
            //    new double[] {1, 0}, new double[] {0, 1},
            //    new double[] {0, 1}, new double[] {1, 0}
            //};

            double[][] inputs =
            {
                //               input         output
                new double[] { 0, 1, 1, 0 }, //  0 
                new double[] { 0, 1, 0, 0 }, //  0
                new double[] { 0, 0, 1, 0 }, //  0
                new double[] { 0, 1, 1, 0 }, //  0
                new double[] { 0, 1, 0, 0 }, //  0
                new double[] { 1, 0, 0, 0 }, //  1
                new double[] { 1, 0, 0, 0 }, //  1
                new double[] { 1, 0, 0, 1 }, //  1
                new double[] { 0, 0, 0, 1 }, //  1
                new double[] { 0, 0, 0, 1 }, //  1
                new double[] { 1, 1, 1, 1 }, //  2
                new double[] { 1, 0, 1, 1 }, //  2
                new double[] { 1, 1, 0, 1 }, //  2
                new double[] { 0, 1, 1, 1 }, //  2
                new double[] { 1, 1, 1, 1 }, //  2
            };

            double[][] outputs = // those are the class labels
            {
                new double[] {1, 0, 0},
                new double[] {1, 0, 0},
                new double[] {1, 0, 0},
                new double[] {1, 0, 0},
                new double[] {1, 0, 0},
                new double[] {0, 1, 0},
                new double[] {0, 1, 0},
                new double[] {0, 1, 0},
                new double[] {0, 1, 0},
                new double[] {0, 1, 0},
                new double[] {0, 0, 1},
                new double[] {0, 0, 1},
                new double[] {0, 0, 1},
                new double[] {0, 0, 1},
                new double[] {0, 0, 1},
            };


            // Setup the deep belief network and initialize with random weights.
            Console.WriteLine(inputs.First().Length);
            DeepBeliefNetwork network = new DeepBeliefNetwork(inputs.First().Length, 2, outputs.First().Length);
            new GaussianWeights(network, 0.1).Randomize();
            network.UpdateVisibleWeights();

            // Setup the learning algorithm.
            DeepBeliefNetworkLearning teacher = new DeepBeliefNetworkLearning(network)
            {
                Algorithm = (h, v, i) => new ContrastiveDivergenceLearning(h, v)
                {
                    LearningRate = 0.1,
                    Momentum = 0.5,
                    Decay = 0.001,
                }
            };

            // Setup batches of input for learning.
            int batchCount = Math.Max(1, inputs.Length / 100);
            // Create mini-batches to speed learning.
            int[] groups = Accord.Statistics.Tools.RandomGroups(inputs.Length, batchCount);
            double[][][] batches = inputs.Subgroups(groups);
            // Learning data for the specified layer.
            double[][][] layerData;

            // Unsupervised learning on each hidden layer, except for the output layer.
            for (int layerIndex = 0; layerIndex < network.Machines.Count - 1; layerIndex++)
            {
                teacher.LayerIndex = layerIndex;
                layerData = teacher.GetLayerInput(batches);
                for (int i = 0; i < 50000; i++)
                {
                    double error = teacher.RunEpoch(layerData) / inputs.Length;
                    //if (i % 10 == 0)
                    //{
                    //    Console.WriteLine(i + ", Error = " + error);
                    //}
                }
            }

            // Supervised learning on entire network, to provide output classification.
            var teacher2 = new Accord.Neuro.Learning.BackPropagationLearning(network)
            {
                LearningRate = 0.1,
                Momentum = 0.5
            };

            // Run supervised learning.
            for (int i = 0; i < 50000; i++)
            {
                double error = teacher2.RunEpoch(inputs, outputs) / inputs.Length;
                //if (i % 10 == 0)
                //{
                //    Console.WriteLine(i + ", Error = " + error);
                //}
            }

            // Test the resulting accuracy.
            //int correct = 0;
            //for (int i = 0; i < inputs.Length; i++)
            //{
            //    double[] outputValues = network.Compute(testInputs[i]);
            //    if (DataManager.FormatOutputResult(outputValues) == DataManager.FormatOutputResult(testOutputs[i]))
            //    {
            //        correct++;
            //    }
            //}

            //Console.WriteLine("Correct " + correct + "/" + inputs.Length + ", " + Math.Round(((double)correct / (double)inputs.Length * 100), 2) + "%");

            //double[] probs = network.GenerateOutput(new double[] { 0, 0 });
            //foreach (double p in probs)
            //{
            //    Console.Write(p + ", ");
            //}
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] output = network.Compute(inputs[i]);
                int imax; output.Max(out imax);
                Console.Write(imax + " -- ");
                foreach (double p in output)
                {
                    Console.Write(p + ", ");
                }
                Console.WriteLine("\n------------------");
            }
        }
    }
}
