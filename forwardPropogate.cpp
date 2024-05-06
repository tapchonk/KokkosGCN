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

#include "forwardPropogate.hpp"

/**
 * @brief Performs forward propagation on a neural network.
 * 
 * This function takes in node embeddings, weights, biases, and other parameters
 * to perform forward propagation on a neural network. It calculates the output
 * of each layer using matrix multiplication and applies activation functions
 * (leaky ReLU) to the intermediate results. The final output is
 * normalized using the softmax function.
 * 
 * @param node_embeddings_layer_1 The input node embeddings for layer 1.
 * @param forward_propogate_results_layer1 The output results for layer 1.
 * @param forward_propogate_results_layer2 The output results for layer 2.
 * @param weights1 The weights for layer 1.
 * @param weights2 The weights for layer 2.
 * @param biases1 The biases for layer 1.
 * @param biases2 The biases for layer 2.
 * @param feature_size The size of the input features.
 * @param neurons_per_layer The number of neurons in each layer.
 * @param num_vertices The number of vertices in the graph.
 * @param start_index The starting index for parallelization.
 * @param hidden_layer_size The size of the hidden layer.
 * @return int Returns 1 upon successful completion.
 */
int ForwardPropogate(KokkosNodeEmbeddingType node_embeddings_layer_1,
                     ForwardPropogateResultsType forward_propogate_results_layer1,
                     ForwardPropogateResultsType forward_propogate_results_layer2,
                     WeightTypeDevice weights1,
                     WeightTypeDevice weights2,
                     BiasTypeDevice biases1,
                     BiasTypeDevice biases2,
                     int feature_size,
                     int neurons_per_layer,
                     int num_vertices,
                     int start_index,
                     int hidden_layer_size) {

  using current_range_policy = Kokkos::RangePolicy<MemSpace::execution_space, Kokkos::Schedule<Kokkos::Static>>;

  int problem_size = num_vertices - start_index;

  using ExecutionSpace = MemSpace::execution_space;
  using Kokkos::parallel_for;
  using Kokkos::TeamPolicy;
  using Kokkos::TeamThreadRange;
  using Kokkos::ThreadVectorRange;
  using Kokkos::Max;
  using Kokkos::exp;

  Kokkos::fence();
  // Forward propogate the neural network to every node embedding
  Kokkos::parallel_for(Kokkos::TeamPolicy<ExecutionSpace>(problem_size, Kokkos::AUTO),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<ExecutionSpace>::member_type& team1) {
      long long i = team1.league_rank() + start_index;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team1, neurons_per_layer),
        [&] (long long j) {
          float neuronOutput = 0.0f;
          Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team1, feature_size),
            [=] (int k, float& temp_sum_inner) {
              temp_sum_inner += node_embeddings_layer_1(i, k) * weights1(k, j);
          }, neuronOutput);
          neuronOutput += biases1(j);
          forward_propogate_results_layer1(i, j) = neuronOutput;
      });

    // Apply a (leaky) relu activation function to the output of the layer
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team1, hidden_layer_size), [=] (long long j) {
        if (forward_propogate_results_layer1(i, j) < 0.0f) {
          forward_propogate_results_layer1(i, j) *= 0.2f;
        }
      });

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team1, neurons_per_layer), [&] (long long j) {
        float neuronOutput = 0.0f;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange (team1, hidden_layer_size), [=] (int k, float& temp_sum_inner) {
          temp_sum_inner += forward_propogate_results_layer1(i, k) * weights2(k, j);
        }, neuronOutput);
        neuronOutput += biases2(j);
        forward_propogate_results_layer2(i, j) = neuronOutput;
      });
  });

  Kokkos::fence();

  Kokkos::parallel_for(Kokkos::TeamPolicy<ExecutionSpace>(problem_size, Kokkos::AUTO),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<ExecutionSpace>::member_type& team2) {

      int i = team2.league_rank() + start_index;

      // before applying the softmax function, we need to
      // normalise the output of the second layer
      // subtract the maximal value from each element

      // this solves the nan problem
      // introduce a small error to the output of the
      // second layer to avoid division by zero
      float max = 0.0000001f;
      
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team2, neurons_per_layer),
        [=] (long long j, float& temp_max) {
          if (forward_propogate_results_layer2(i, j) > temp_max) {
            temp_max = forward_propogate_results_layer2(i, j);
          }
      },  Kokkos::Max<float>(max));

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team2, neurons_per_layer),
        [=] (long long j) {
          forward_propogate_results_layer2(i, j) -= max;
      });

    // Apply a softmax activation function to the output of the second layer
      float sum = 0.0f;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team2, neurons_per_layer),
        [=] (long long j, float& temp_sum) {
          forward_propogate_results_layer2(i, j) = Kokkos::exp(forward_propogate_results_layer2(i, j));
          temp_sum += forward_propogate_results_layer2(i, j);
      }, sum);

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team2, neurons_per_layer),
        [=] (long long j) {
          forward_propogate_results_layer2(i, j) = forward_propogate_results_layer2(i, j) / sum;
      } );
  });

  Kokkos::fence();


  return 1;
}
