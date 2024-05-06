#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <utility>
#include <stdlib.h>

#include "initialiseWeights.hpp"

/**
 * @brief Initialise the weights and biases of the neural network
 * The weights are initialised using a normal distribution with a mean of 0 and a standard deviation of 0.125 and -0.125 from the mean.
 * We then perform the absolute function to ensure that the weights are positive.
 * Our biases are initialised to 0.
 * 
 * @param weights1: The weights matrix for the first layer of the neural network.
 * @param weights2: The weights matrix for the second layer of the neural network.
 * @param feature_size: The size of the input features.
 * @param neurons_per_layer: The number of neurons in each layer of the neural network.
 * @param hidden_layer_size: The size of the hidden layer in the neural network.
 * @param num_threads: The number of threads used for parallel execution.
 * @return int: Returns 1 upon successful initialization.
 */
int InitialiseWeights(WeightType1 weights1,
                      WeightType1 weights2,
                      int feature_size,
                      int neurons_per_layer,
                      int hidden_layer_size,
                      int num_threads        ) {


  int seed = 42;
  // Set the seed for the random number generator
  std::random_device rd;
  //Seed the random number generator for consistent results
  std::mt19937 gen(seed); // Set the seed value here
  std::normal_distribution<> d(-0.125f, 0.125f);
  //std::normal_distribution<> d(0.0, 0.25);

  for (int i = 0; i < feature_size; ++i) {
    for (int j = 0; j < hidden_layer_size; ++j) {
      weights1(i, j) = abs(d(gen));
    }
  }

  for (int i = 0; i < hidden_layer_size; ++i) {
    for (int j = 0; j < neurons_per_layer; ++j) {
      weights2(i, j) = abs(d(gen));
    }
  }
  
  return 1;
}

/**
 * @brief Initialise the expected results of the neural network.
 * This function sets the expected results of the neural network to 1 if the node label is equal to the index of the expected results and 0 otherwise.
 * This is for the classification task that utilises one hot encoding.
 * 
 * @param expected_results 
 * @param node_labels 
 * @param num_vertices 
 * @param neurons_per_layer 
 * @return int 
 */
int InitialiseExpected (ExpectedType1 expected_results,
                        NodeLabelsType node_labels,
                        long long num_vertices,
                        int neurons_per_layer) {
  // Initialise the expected results of the neural network
  for (int i = 0; i < num_vertices; i++) {
    for (int j = 0; j < neurons_per_layer; ++j) {
      if (node_labels(i) == j)
        expected_results(i, j) = 1.0f;
      else
        expected_results(i, j) = 0.0f;
    }
  }

  return 1;
}