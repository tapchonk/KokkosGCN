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
#include <iostream>

#include "backPropogate.hpp"
#include <Kokkos_Core.hpp>

/**
 * @brief Performs backpropagation on a neural network.
 * 
 * This function takes in node embeddings, forward propogate results, expected results, weights, biases, and other parameters
 * to perform backpropagation on a neural network. It calculates the error of the output layer and the hidden layer
 * and updates the weights and biases accordingly.
 * 
 * The current loss function is the cross entropy loss.
 * The following code has been multithreaded using Kokkos to run optimised parallel code on the GPU and on x86_64 CPUs.
 * 
 * @param node_embeddings_layer1 The input node embeddings for layer 1.
 * @param forward_propagate_results_layer1 The output results for layer 1.
 * @param forward_propagate_results_layer2 The output results for layer 2.
 * @param expected_results The expected results for the output layer.
 * @param weights1 The weights for layer 1.
 * @param weights2 The weights for layer 2.
 * @param gradients_layer1 The gradients for layer 1.
 * @param gradients_layer2 The gradients for layer 2.
 * @param biases1 The biases for layer 1.
 * @param biases2 The biases for layer 2.
 * @param feature_size The size of the input features.
 * @param neurons_per_layer The number of neurons in each layer.
 * @param num_vertices The number of vertices in the graph.
 * @param learning_rate The learning rate for the neural network.
 * @param batch_size The size of the batch for backpropagation.
 * @param num_threads The number of threads used for parallel execution.
 * @param hidden_layer_size The size of the hidden layer.
 * @return void
 */
void BackpropagateBatch(KokkosNodeEmbeddingTypeTranspose node_embeddings_layer1_transpose,
                        ForwardPropogateResultsType forward_propagate_results_layer1,
                        ForwardPropogateResultsType forward_propagate_results_layer2,
                        ForwardPropogateResultsTypeTranspose forward_propagate_results_layer1_transpose,
                        ExpectedResultsType expected_results,
                        WeightTypeDevice weights1,
                        WeightTypeDevice weights2,
                        GradientsTypeDevice gradients_layer1,
                        GradientsTypeDevice gradients_layer2,
                        GradientsTypeDeviceTranspose gradients_layer1_transpose,
                        GradientsTypeDeviceTranspose gradients_layer2_transpose,
                        BiasTypeDevice biases1,
                        BiasTypeDevice biases2,
                        int feature_size,
                        int neurons_per_layer,
                        int num_vertices,
                        float learning_rate,
                        int batch_size,
                        int num_threads,
                        int hidden_layer_size) {

  using current_range_policy = Kokkos::TeamPolicy<MemSpace::execution_space,  Kokkos::Schedule<Kokkos::Static>>;
  using dynamic_range_policy = Kokkos::RangePolicy<MemSpace::execution_space, Kokkos::Schedule<Kokkos::Static>>;

  using Kokkos::parallel_for;
  using Kokkos::parallel_reduce;
  using Kokkos::TeamThreadRange;
  using Kokkos::ThreadVectorRange;
  using Kokkos::TeamVectorRange;
  using Kokkos::single;
  using Kokkos::PerTeam;
  using Kokkos::atomic_fetch_add;

  std::chrono::duration<double> loop_1_time;
  std::chrono::duration<double> loop_2_time;
  std::chrono::duration<double> loop_3_time;
  std::chrono::duration<double> loop_4_time;
  std::chrono::duration<double> loop_5_time;
  std::chrono::duration<double> loop_6_time;
  std::chrono::duration<double> loop_7_time;
  std::chrono::duration<double> loop_8_time;
  std::chrono::duration<double> loop_9_time;
  std::chrono::duration<double> loop_10_time;

  #if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int numSMs = deviceProp.multiProcessorCount;

    // Set the number of threads for the nested parallelism
    int NESTED_THREADS_L1 = numSMs;
    int NESTED_THREADS_L2 = numSMs;
  #else
    int NESTED_THREADS_L1 = 72;
    int NESTED_THREADS_L2 = 72;
  #endif

  // timers
  auto start = std::chrono::high_resolution_clock::now();

  Kokkos::fence();

  // Backpropagate the neural network in full batch mode where the batch size is equal to the number of vertices
  parallel_for(current_range_policy(num_vertices, Kokkos::AUTO),
    KOKKOS_LAMBDA (const current_range_policy::member_type& team1) {
      long long i = team1.league_rank();
      // Calculate the error of the output layer
      parallel_for(TeamThreadRange(team1, neurons_per_layer), [=] (long long j) {
          gradients_layer2(i, j) = (forward_propagate_results_layer2(i, j) - expected_results(i, j));
      });

      // Calculate the error of the hidden layer
      parallel_for(TeamThreadRange(team1, hidden_layer_size), [=] (long long j) {
          float sum = 0.0f;
          parallel_reduce(ThreadVectorRange(team1, neurons_per_layer), [=] (int k, float& temp_sum) {
              temp_sum += gradients_layer2(i, k) * weights2(j, k);
          }, sum);
          gradients_layer1(i, j) = sum;
      });
  });

  Kokkos::fence();

  auto end = std::chrono::high_resolution_clock::now();

  loop_1_time = end - start;

  start = std::chrono::high_resolution_clock::now();
  
  // ridiculously fast way to transpose matrices

  const int block_size = 32; // Set the block size for loop blocking

  parallel_for(current_range_policy(num_vertices / block_size, Kokkos::AUTO),
    KOKKOS_LAMBDA (const current_range_policy::member_type& team1) {
      int block_start = team1.league_rank() * block_size;
      // catch vertices if num_vertices is not divisible by block_size
      int catch_vertices = 0;
      if (team1.league_rank() + 1 == num_vertices / block_size) catch_vertices = num_vertices % block_size;

      parallel_for(TeamThreadRange(team1, block_size + catch_vertices), [=] (int i) {
        i += block_start;
        parallel_for(ThreadVectorRange(team1, hidden_layer_size), [=] (int j) {
          gradients_layer1_transpose(i, j) = gradients_layer1(i, j);
        });
      });
  });

  Kokkos::fence();

  end = std::chrono::high_resolution_clock::now();

  loop_2_time = end - start;

  start = std::chrono::high_resolution_clock::now();

  parallel_for(current_range_policy(num_vertices / block_size, Kokkos::AUTO),
    KOKKOS_LAMBDA (const current_range_policy::member_type& team1) {
      int block_start = team1.league_rank() * block_size;
      // catch vertices if num_vertices is not divisible by block_size
      int catch_vertices = 0;
      if (team1.league_rank() + 1 == num_vertices / block_size) catch_vertices = num_vertices % block_size;

      parallel_for(TeamThreadRange(team1, block_size + catch_vertices), [=] (int i) {
        i += block_start;
        parallel_for(ThreadVectorRange(team1, neurons_per_layer), [=] (int j) {
          gradients_layer2_transpose(i, j) = gradients_layer2(i, j);
        });
      });
  });

  Kokkos::fence();

  end = std::chrono::high_resolution_clock::now();

  loop_3_time = end - start;

  start = std::chrono::high_resolution_clock::now();

  // transpose forward propogate results
  parallel_for(current_range_policy(num_vertices / block_size, Kokkos::AUTO),
    KOKKOS_LAMBDA (const current_range_policy::member_type& team1) {
      int block_start = team1.league_rank() * block_size;
      // catch vertices if num_vertices is not divisible by block_size
      int catch_vertices = 0;
      if (team1.league_rank() + 1 == num_vertices / block_size) catch_vertices = num_vertices % block_size;

      parallel_for(TeamThreadRange(team1, block_size + catch_vertices), [=] (int i) {
        i += block_start;
        parallel_for(ThreadVectorRange(team1, hidden_layer_size), [=] (int j) {
          forward_propagate_results_layer1_transpose(i, j) = forward_propagate_results_layer1(i, j);
        });
      });
  });

  Kokkos::fence();

  end = std::chrono::high_resolution_clock::now();

  loop_4_time = end - start;

  start = std::chrono::high_resolution_clock::now();

  long long total_updates_L2 = hidden_layer_size * neurons_per_layer * NESTED_THREADS_L1;

  // Kokkos::AUTO decides the thread launch configuration based on the number of threads
  parallel_for(current_range_policy(total_updates_L2, Kokkos::AUTO),
    KOKKOS_LAMBDA (const current_range_policy::member_type& team1) {

      int rank = team1.league_rank();
      int start = (num_vertices * (rank / (hidden_layer_size * neurons_per_layer))) / NESTED_THREADS_L1;

      rank = rank % (hidden_layer_size * neurons_per_layer);

      int j = rank % neurons_per_layer;
      int k = rank / neurons_per_layer;

      float sum_outer = 0.0f;

      int loop_size = num_vertices / NESTED_THREADS_L1;

      if (start == (num_vertices / NESTED_THREADS_L1)*(NESTED_THREADS_L1 - 1))
        loop_size += (num_vertices % NESTED_THREADS_L1);

      // Decompose the reduction operations into multiple smaller reductions to saturate memory bandwidth
      parallel_reduce(TeamThreadRange(team1, loop_size),
        [=] (int i, float& temp_sum_outer) {
            temp_sum_outer -= gradients_layer2_transpose(i + start, j) * forward_propagate_results_layer1_transpose(i + start, k);
      }, sum_outer);

      // Maintains atomicity for the weights update
      team1.team_barrier();

      // Only one thread update per team
      single(PerTeam(team1), [=] () {
      //if (team1.team_rank() == 0)
        atomic_fetch_add(&weights2(k, j), sum_outer * learning_rate);
      });
  });

  Kokkos::fence();

  end = std::chrono::high_resolution_clock::now();

  loop_5_time = end - start;

  Kokkos::fence();

  start = std::chrono::high_resolution_clock::now();
  
  long long total_updates_L1 = feature_size * hidden_layer_size * NESTED_THREADS_L2;

  parallel_for(dynamic_range_policy(0, num_vertices * hidden_layer_size), KOKKOS_LAMBDA (const long long i) {
      long long j = i / num_vertices;
      long long k = i % num_vertices;
      if (forward_propagate_results_layer1_transpose(k, j) <= 0.0f) gradients_layer1_transpose(k, j) *= 0.2f;

  });

  Kokkos::fence();

  end = std::chrono::high_resolution_clock::now();

  loop_6_time = end - start;

  start = std::chrono::high_resolution_clock::now();
  
  parallel_for(current_range_policy(total_updates_L1, Kokkos::AUTO),
    KOKKOS_LAMBDA (const current_range_policy::member_type& team2) {

      long long rank = team2.league_rank();
      int start = (num_vertices * (rank / (hidden_layer_size * feature_size))) / NESTED_THREADS_L2;

      rank = rank % (hidden_layer_size * feature_size);

      long long j = rank % hidden_layer_size;
      long long k = rank / hidden_layer_size;

      float sum_outer = 0.0f;

      long long loop_size = num_vertices / NESTED_THREADS_L2;

      if (start == (num_vertices / NESTED_THREADS_L2)*(NESTED_THREADS_L2 - 1)) loop_size += (num_vertices % NESTED_THREADS_L2);

      parallel_reduce(TeamVectorRange(team2, loop_size),
        [=] (long long i, float& temp_sum_outer) {
          temp_sum_outer -= gradients_layer1_transpose(i + start, j) * node_embeddings_layer1_transpose(i + start, k);
      }, sum_outer);

      team2.team_barrier();
      single(PerTeam(team2), [=] () {
      //if (team2.team_rank() == 0)
        atomic_fetch_add(&weights1(k, j), sum_outer * learning_rate);
      });
  });

  Kokkos::fence();

  end = std::chrono::high_resolution_clock::now();

  loop_7_time = end - start;
  
  Kokkos::fence();

  start = std::chrono::high_resolution_clock::now();
 
  parallel_for(current_range_policy(neurons_per_layer, Kokkos::AUTO),
    KOKKOS_LAMBDA (const current_range_policy::member_type& team3) {
      int k = team3.league_rank();
      float temp_sum = 0.0f;
      parallel_reduce(TeamThreadRange(team3, num_vertices / NESTED_THREADS_L1), [=] (int i, float& sum) {
        float sum_inner = 0.0f;
        parallel_reduce(ThreadVectorRange(team3, NESTED_THREADS_L1), [=] (int l, float& temp_sum_inner) {
          temp_sum_inner -= gradients_layer1_transpose(l + i * NESTED_THREADS_L1, k);
        }, sum_inner);
        sum += sum_inner;
      }, temp_sum);
      team3.team_barrier();
      biases1(k) += temp_sum * learning_rate;
  });

  Kokkos::fence();

  end = std::chrono::high_resolution_clock::now();

  loop_8_time = end - start;

  start = std::chrono::high_resolution_clock::now();

  parallel_for(current_range_policy(neurons_per_layer, Kokkos::AUTO),
    KOKKOS_LAMBDA (const current_range_policy::member_type& team4) {
      int k = team4.league_rank();
      float temp_sum = 0.0f;
      parallel_reduce(TeamThreadRange(team4, num_vertices / NESTED_THREADS_L2), [=] (int i, float& sum) {
        float sum_inner = 0.0f;
        parallel_reduce(ThreadVectorRange(team4, NESTED_THREADS_L2), [=] (int l, float& temp_sum_inner) {
          temp_sum_inner -= gradients_layer2_transpose(l + i * NESTED_THREADS_L2, k);
        }, sum_inner);
        sum += sum_inner;
      }, temp_sum);
      team4.team_barrier();
      biases2(k) += temp_sum * learning_rate;
  });

  Kokkos::fence();

  end = std::chrono::high_resolution_clock::now();

  loop_9_time = end - start;

  start = std::chrono::high_resolution_clock::now();

  // catch rest of workload
  if (num_vertices % NESTED_THREADS_L1 != 0) {
    parallel_for(current_range_policy(neurons_per_layer, Kokkos::AUTO),
      KOKKOS_LAMBDA (const current_range_policy::member_type& team3) {
        int k = team3.league_rank();
        float temp_sum = 0.0f;
        parallel_reduce(TeamThreadRange(team3, num_vertices % NESTED_THREADS_L1), [=] (int l, float& temp_sum_inner) {
          temp_sum_inner -= gradients_layer1_transpose(l + (num_vertices / NESTED_THREADS_L1) * NESTED_THREADS_L1, k);
        }, temp_sum);
        biases1(k) += temp_sum * learning_rate;
    });

    parallel_for(current_range_policy(neurons_per_layer, Kokkos::AUTO),
      KOKKOS_LAMBDA (const current_range_policy::member_type& team4) {
        int k = team4.league_rank();
        float temp_sum = 0.0f;
        parallel_reduce(TeamThreadRange(team4, num_vertices % NESTED_THREADS_L2), [=] (int l, float& temp_sum_inner) {
          temp_sum_inner -= gradients_layer2_transpose(l + (num_vertices / NESTED_THREADS_L2) * NESTED_THREADS_L2, k);
        }, temp_sum);
        biases2(k) += temp_sum * learning_rate;
    });
  }

  Kokkos::fence();

  end = std::chrono::high_resolution_clock::now();

  loop_10_time = end - start;

  std::cout << "Loop 1 time: " << loop_1_time.count() << "s" << std::endl;
  std::cout << "Loop 2 time: " << loop_2_time.count() << "s" << std::endl;
  std::cout << "Loop 3 time: " << loop_3_time.count() << "s" << std::endl;
  std::cout << "Loop 4 time: " << loop_4_time.count() << "s" << std::endl;
  std::cout << "Loop 5 time: " << loop_5_time.count() << "s" << std::endl;
  std::cout << "Loop 6 time: " << loop_6_time.count() << "s" << std::endl;
  std::cout << "Loop 7 time: " << loop_7_time.count() << "s" << std::endl;
  std::cout << "Loop 8 time: " << loop_8_time.count() << "s" << std::endl;
  std::cout << "Loop 9 time: " << loop_9_time.count() << "s" << std::endl;
  std::cout << "Loop 10 time: " << loop_10_time.count() << "s" << std::endl;
  return;
}
