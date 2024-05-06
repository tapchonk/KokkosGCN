/**
 * @file gnnAlgorithm.cpp
 * @brief This is the main file for the GCNN algorithm. It reads in the command line arguments and then runs the GCNN algorithm.
 * The GCNN currently uses the following algorithm: 
 * Performs the following convolutions on the graph:
 * 
 * 1. Generate the node embeddings for the graph.
 * 2. Create the neural network and apply it to the node embeddings.
 * 3. Backpropagate the neural network to update the weights and biases.
 * 4. Forward propogate the neural network to get the accuracy and loss.
 * 5. Repeat steps 2-4 for the number of epochs given by the user.
 * 6. Forward propogate the neural network to get the accuracy and loss for the test data.
 * 7. Print the results.
 * 
 * The current neural network architecture is the following:
 * 100 -> 47 -> Leaky ReLU Activation Function -> 47 Softmax Activation Function
 * 
 * The algorithm uses the Cross Entropy Loss as the loss function and the MSE as the accuracy function.
 * We specify the convolution function as averaging the node embeddings of the neighbours of each node to neighbours one hop from each node.
 * We then perform the aggregation function on that updated node embedding to get the new node embedding on the two hop neighbours.
 * We can perform this aggregation function for as many convolutions as we want.
 * 
 * 
 * <KOKKOS IMPLEMENTATION OF THE CODE>
 */
// ./remoterun.sh ./gnnAlgorithm.cuda -Train 2000000 -Test 449029 -Hidden 47 -E 1000 -C 3 -LR 0.2f -AT 80.0f
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology  Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

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
#include <iomanip>
#include <iostream>

#ifdef KOKKOS_ENABLE_CUDA
  #include <cuda_runtime.h>
  #include <cudnn.h>
#endif


#include "checkSizes.h"
#include "getGraphSize.hpp"
#include "readGraphData.hpp"
#include "initialiseWeights.hpp"
#include "accuracyErrorUtil.hpp"
#include "generateNodeEmbeddings.hpp"
#include "forwardPropogate.hpp"
#include "backPropogate.hpp"

#ifdef USING_SILO
  #include <silo.h>
  #include "writeTimestep.h"
#endif

#ifdef USING_THRUST
  #include <thrust/sort.h>
  #include <thrust/execution_policy.h>
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_NestedSort.hpp>
#include <Kokkos_StaticCrsGraph.hpp>

/**
 * @brief The following program is the starting point for the GNN algorithm. It reads in the command line arguments and then runs the GNN algorithm.
 * The current neural network architecture is a 1 input layer, 1 hidden layer and 1 output layer. The input layer has the same number of neurons as the number of features in the graph.
 * The hidden layer has a user defined number of neurons and the output layer has the same number of neurons as the number of classes in the graph (47).
 * 
 * Hence the neural network architecture is as follows:
 * 100 -> 47 -> Leaky ReLU Activation Function -> 47 Softmax Activation Function -> Output with one hot encoding
 * 
 * The algorithm uses the mean square error as the loss function and the accuracy as the accuracy function.
 * The algorithm uses the backpropagation algorithm to update the weights and biases of the neural network.
 * 
 * <THIS IS THE KOKKOS VERSION OF THE GNN ALGORITHM>
 * 
 * @param Train 
 * @param Test
 * @param Hidden
 * @param Epochs
 * @param Convolutions
 * @param Learning_Rate
 * @param Accuracy_Threshold
 * 
 * @return int 
 */
int main( int argc, char* argv[] )
{
  long long train_size = -1;
  long long test_size = -1;
  long long hidden_layer_size = -1; // nice
  long long num_epochs = -1;
  long long convolutions = -1;
  float learning_rate = -0.0f;
  float accuracy_threshold = -0.0f;

  // Timers for the algorithm
  std::chrono::duration<double> initialisationTime;
  std::chrono::duration<double> generateNodeEmbeddingTime;
  std::chrono::duration<double> neuralNetworkTime;
  std::chrono::duration<double> backpropagateTime;
  std::chrono::duration<double> forwardpropagateTime;
  std::chrono::duration<double> utilityTime;

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-Train" ) == 0 ) || ( strcmp( argv[ i ], "-Train Size" ) == 0 ) ) {
      train_size = atoi( argv[ ++i ] );
      printf( "  User Train Size is %ld\n", train_size );
    }
    if ( ( strcmp( argv[ i ], "-Test" ) == 0 ) || ( strcmp( argv[ i ], "-Test Size" ) == 0 ) ) {
      test_size = atoi( argv[ ++i ] );
      printf( "  User Test Size is %ld\n", test_size );
    }
    if ( ( strcmp( argv[ i ], "-Hidden" ) == 0 ) || ( strcmp( argv[ i ], "-Hidden Layer Size" ) == 0 ) ) {
      hidden_layer_size = atoi( argv[ ++i ] );
      printf( "  User Hidden Layer Size is %ld\n", hidden_layer_size );
    }
    if ( ( strcmp( argv[ i ], "-E" ) == 0 ) || ( strcmp( argv[ i ], "-Epochs" ) == 0 ) ) {
      num_epochs = atoi( argv[ ++i ] );
      printf( "  User Number of Epochs is %ld\n", num_epochs );
    }
    if ( ( strcmp( argv[ i ], "-C" ) == 0 ) || ( strcmp( argv[ i ], "-Convolutions" ) == 0 ) ) {
      convolutions = atoi( argv[ ++i ] );
      printf( "  User Number of Convolutions is %ld\n", convolutions );
    }
    if ( ( strcmp( argv[ i ], "-LR" ) == 0 ) || ( strcmp( argv[ i ], "-Learning Rate" ) == 0 ) ) {
      learning_rate = atof( argv[ ++i ] );
      printf( "  User Learning Rate is %f\n", learning_rate );
    }
    if ( ( strcmp( argv[ i ], "-AT" ) == 0 ) || ( strcmp( argv[ i ], "-Accuracy Threshold" ) == 0 ) ) {
      accuracy_threshold = atof( argv[ ++i ] );
      printf( "  User Accuracy Threshold is %f\n", accuracy_threshold );
    }


    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      fprintf(stdout,  "  GNN Options:\n" );
      fprintf(stdout,  "  -Train Size        (-Train )  <long>:   num, determines number of training data points (num) (default: 800)\n" );
      fprintf(stdout,  "  -Test Size         (-Test )   <long>:   num, determines number of testing data points (num) (default: 200)\n" );
      fprintf(stdout,  "  -Hidden Layer Size (-Hidden ) <long>:   num, determines number of neurons in the hidden layer (num) (default: 47)\n" );
      fprintf(stdout,  "  -Epochs            (-E )      <long>:   num, determines the number of epochs (num) (default: 100)\n" );
      fprintf(stdout,  "  -Convolutions      (-C )      <long>:   num, determines the number of convolutions (num) (default: 1)\n" );
      fprintf(stdout,  "  -Learning Rate     (-LR )     <float>:  num, determines the learning rate (num) (default: 0.1)\n" );
      fprintf(stdout,  "  -Accuracy Threshold(-AT )     <float>:  num, determines the accuracy threshold (num) (default: 100.1)\n" );
      fprintf(stdout,  "  -help              (-h ):         print this message\n\n" );
      exit( 1 );
    }
  }

  //Error check the sizes given by the user
  checkSizes(train_size, test_size, hidden_layer_size, num_epochs, convolutions, learning_rate, accuracy_threshold);

  Kokkos::initialize( argc, argv );
  {

  int num_threads;

  // Set the number of threads to be used by the algorithm from the OpenMP environment variable
  if (getenv("OMP_NUM_THREADS"))
    num_threads = std::atoi(getenv("OMP_NUM_THREADS"));
  fprintf(stdout, "\n================================================================================\n");
  fprintf(stdout, "Number of Threads is : %d.\n", num_threads);
  fprintf(stdout, "\n================================================================================\n");

  
  #if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
    #define MemSpace Kokkos::CudaSpace
    #define MemLayout Kokkos::LayoutRight
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    fprintf(stdout, "<ONLY APPLICABLE TO AMPERE ADA GPUS>  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           128,
           128 *
               deviceProp.multiProcessorCount);
    num_threads = 128 * deviceProp.multiProcessorCount;
  #endif
  #if !defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
    #define MemSpace Kokkos::OpenMP
    #define MemLayout Kokkos::LayoutRight
    using openmp_range_policy = Kokkos::RangePolicy<Kokkos::OpenMP::execution_space>;
  #endif
  #ifdef KOKKOS_ENABLE_HIP // (if we want to add support for Radeon GPUs later)
    #define MemSpace Kokkos::Experimental::HIPSpace
    #define MemLayout Kokkos::LayoutRight
  #endif
  #ifdef KOKKOS_ENABLE_SERIAL
    #define MemSpace Kokkos::Serial
    #define MemLayout Kokkos::LayoutRight
  #endif

  using ExecutionSpace = MemSpace::execution_space;
  using StaticCrsGraphType = Kokkos::StaticCrsGraph<int, Kokkos::LayoutRight, MemSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>, int>;

  // Start timers for the algorithm
  auto start = std::chrono::high_resolution_clock::now();

  long long num_edges;
  long long num_vertices;
  long long feature_size;
  
  // 2449029
  int start_test_index = train_size;
  int batch_size = 0;
  int neurons_per_layer = 47;

  // File paths for the graph data from the products dataset
  std::string vertex_size_file = std::string("../products/raw/num-node-list.csv");
  std::string edge_size_file = std::string("../products/raw/num-edge-list.csv");
  std::string edge_file = std::string("../products/raw/edge.csv");
  std::string feature_file = std::string("../products/raw/node-feat.csv");
  std::string label_file = std::string("../products/raw/node-label.csv");


  //Read in the graph size data from the files
  if (getGraphSize(vertex_size_file, edge_size_file, feature_file, &num_vertices, &num_edges, &feature_size) != 0) {
    fprintf(stdout, "Failed to read in the graph size data.\n");
    return -1;
  }

  //transpose the node features and forward propogate results 1
  Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace> node_features_transpose("node_features_transpose", num_vertices, feature_size);
  Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace> forward_propogate_results_layer1_transpose("forward_node_labels_test_layer_1_transpose", num_vertices, hidden_layer_size);

  //transpose the gradients
  Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace> gradients_layer1_transpose("gradients_layer1_transpose", num_vertices, feature_size);
  Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace> gradients_layer2_transpose("gradients_layer2_transpose", num_vertices, neurons_per_layer);

  Kokkos::View<float*, MemLayout, MemSpace> biases1("biases1", hidden_layer_size);
  Kokkos::View<float*, MemLayout, MemSpace> biases2("biases2", neurons_per_layer);

  // For the gradients
  Kokkos::View<float**, MemLayout, MemSpace> gradients_layer1("gradients_layer1", num_vertices, feature_size);
  Kokkos::View<float**, MemLayout, MemSpace> gradients_layer2("gradients_layer2", num_vertices, neurons_per_layer);

  Kokkos::View<float**, MemLayout, MemSpace> node_features("node_features", num_vertices, feature_size);
  Kokkos::View<float**, MemLayout, MemSpace> node_embeddings_layer_1("node_embeddings_layer_1", num_vertices, feature_size);
  Kokkos::View<float**, MemLayout, MemSpace> node_embeddings_layer_2("node_embeddings_layer_2", num_vertices, feature_size);
  Kokkos::View<int*, MemSpace> node_labels("node_labels", num_vertices);

  // Result store and utility
  Kokkos::View<float**, MemLayout, MemSpace> forward_propogate_results_layer1("forward_node_labels_test_layer_1", num_vertices, hidden_layer_size);
  Kokkos::View<float**, MemLayout, MemSpace> forward_propogate_results_layer2("forward_node_labels_test_layer_2", num_vertices, neurons_per_layer);
  Kokkos::View<float**, MemLayout, MemSpace> expected_results("expected_results", num_vertices, neurons_per_layer);

  // For the neural network
  Kokkos::View<float**, MemLayout, MemSpace> weights1("weights1", feature_size, hidden_layer_size);
  Kokkos::View<float**, MemLayout, MemSpace> weights2("weights2", hidden_layer_size, neurons_per_layer);

  #ifdef KOKKOS_ENABLE_CUDA
    // Create Host mirrors of the views
    //for the gradients
    Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_gradients_layer1 = Kokkos::create_mirror_view(gradients_layer1);
    Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_gradients_layer2 = Kokkos::create_mirror_view(gradients_layer2);

    Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_node_features = Kokkos::create_mirror_view(node_features);
    Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_node_embeddings_layer_1 = Kokkos::create_mirror_view(node_embeddings_layer_1);
    Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_node_embeddings_layer_2 = Kokkos::create_mirror_view(node_embeddings_layer_2);
    Kokkos::View<int*, MemSpace>::HostMirror h_node_labels = Kokkos::create_mirror_view(node_labels);

    Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_forward_propogate_results_layer1 = Kokkos::create_mirror_view(forward_propogate_results_layer1);
    Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_forward_propogate_results_layer2 = Kokkos::create_mirror_view(forward_propogate_results_layer2);
    Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_expected_results = Kokkos::create_mirror_view(expected_results);

    Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_weights1 = Kokkos::create_mirror_view(weights1);
    Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_weights2 = Kokkos::create_mirror_view(weights2);

  #endif

  // Output the graph size data
  fprintf(stdout, "<Number of vertices: %d>\n", num_vertices);
  fprintf(stdout, "<Number of edges: %d>\n", num_edges);
  fprintf(stdout, "<Feature size: %d>\n", feature_size);

  // For the graph structure
  std::vector<std::vector<int>> graph_edges(num_vertices * 2);

  // Initialise Cuda memory spaces for the graph data and performs copies between the host device and the GPU device
  #if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
    // Read in the graph data from the files
    readGraphData(label_file, feature_file, edge_file, h_node_labels, h_node_features, &graph_edges, num_vertices, num_edges, feature_size);
    InitialiseWeights(h_weights1, h_weights2, feature_size, neurons_per_layer, hidden_layer_size, num_threads);
    InitialiseExpected(h_expected_results, h_node_labels, num_vertices, neurons_per_layer);
    Kokkos::deep_copy(node_labels, h_node_labels);
    Kokkos::deep_copy(node_features, h_node_features);
    Kokkos::deep_copy(weights1, h_weights1);
    Kokkos::deep_copy(weights2, h_weights2);
    Kokkos::deep_copy(expected_results, h_expected_results);
  #endif
  #if (!defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)) || defined(KOKKOS_ENABLE_SERIAL)
    readGraphData(label_file, feature_file, edge_file, node_labels, node_features, &graph_edges, num_vertices, num_edges, feature_size);
    InitialiseWeights(weights1, weights2, feature_size, neurons_per_layer, hidden_layer_size, num_threads);
    InitialiseExpected(expected_results, node_labels, num_vertices, neurons_per_layer);
  #endif

  // Create the graph structure for the graph
  StaticCrsGraphType d_graph;
  d_graph = Kokkos::create_staticcrsgraph<StaticCrsGraphType>("d_graph", graph_edges);

  auto end = std::chrono::high_resolution_clock::now();

  //Store the time it took to run the function
  initialisationTime += end - start;
  
  start = std::chrono::high_resolution_clock::now();

  // Generate the node embeddings for the graph
  for (long long i = 0; i < convolutions; i++) {
    if (i == 0) {
      GenerateNodeEmbeddingLayer(d_graph, node_features, node_embeddings_layer_1, 0, feature_size, num_vertices);
      node_features = node_embeddings_layer_1;
    }
    // Pass the node embeddings between layer 1/2 and 2/1 dependent on the iteration
    else {
      if (i % 2 == 0) {
        GenerateNodeEmbeddingLayer(d_graph, node_embeddings_layer_2, node_embeddings_layer_1, 0, feature_size, num_vertices);
        node_features = node_embeddings_layer_1;
      }
      else {
        GenerateNodeEmbeddingLayer(d_graph, node_embeddings_layer_1, node_embeddings_layer_2, 0, feature_size, num_vertices);
        node_features = node_embeddings_layer_2;
      }
    }
  }

  //make new graph
  StaticCrsGraphType dealloc_graph;

  // deallocate the graph
  d_graph = dealloc_graph;

  Kokkos::fence();

  end = std::chrono::high_resolution_clock::now();

  //Store the time it took to run the function
  generateNodeEmbeddingTime += end - start;

  // transpose the node features and store in node_features_transpose
  Kokkos::parallel_for("transpose_node_features", Kokkos::RangePolicy<ExecutionSpace>(0, num_vertices), KOKKOS_LAMBDA (long long i) {
    for (int j = 0; j < feature_size; j++) {
      node_features_transpose(i, j) = node_features(i, j);
    }
  });

  Kokkos::fence();

  fprintf(stdout, "\n||========================================================RESULTS========================================================\n");

  start = std::chrono::high_resolution_clock::now();

  // Create the neural network and apply it to the node embeddings
  float accel = 1.0f;
  float max_accuracy = 0.0f;

  // Set the maximum loss to be a large number
  float max_loss = 1.0E34f;
  
  // Output training results for the neural network
  std::chrono::time_point<std::chrono::high_resolution_clock> forwards_start;
  std::chrono::time_point<std::chrono::high_resolution_clock> forwards_end;
  std::chrono::time_point<std::chrono::high_resolution_clock> utility_start;
  std::chrono::time_point<std::chrono::high_resolution_clock> utility_end;
  std::chrono::time_point<std::chrono::high_resolution_clock> backpropagate_start;
  std::chrono::time_point<std::chrono::high_resolution_clock> backpropagate_end;

  for (int e = 0; e < num_epochs; e++) {

    forwards_start = std::chrono::high_resolution_clock::now();

    // Perform prediction via forward propogation on the neural network
    ForwardPropogate(node_features,
                     forward_propogate_results_layer1,
                     forward_propogate_results_layer2,
                     weights1,
                     weights2,
                     biases1,
                     biases2,
                     feature_size,
                     neurons_per_layer,
                     train_size,
                     0,
                     hidden_layer_size         );
    Kokkos::fence();

    forwards_end = std::chrono::high_resolution_clock::now();

    forwardpropagateTime += forwards_end - forwards_start;

    utility_start = std::chrono::high_resolution_clock::now();

    // Calculate the loss and accuracy of the neural network
    float loss = MeanSquareError(forward_propogate_results_layer2, expected_results, train_size, neurons_per_layer);

    Kokkos::fence();

    float accuracy = CalculateAccuracy(forward_propogate_results_layer2, node_labels, train_size, 0, neurons_per_layer);

    // Epsilon or our training rate is dependent on the size of the training data
    float epsilon = (learning_rate/(float)train_size) * accel;

    Kokkos::fence();

    utility_end = std::chrono::high_resolution_clock::now();

    utilityTime += utility_end - utility_start;

    backpropagate_start = std::chrono::high_resolution_clock::now();

    // Backpropogate the error through the neural network to update the weights and biases
    BackpropagateBatch(node_features_transpose,
                       forward_propogate_results_layer1,
                       forward_propogate_results_layer2,
                       forward_propogate_results_layer1_transpose,
                       expected_results,
                       weights1,
                       weights2,
                       gradients_layer1,
                       gradients_layer2,
                       gradients_layer1_transpose,
                       gradients_layer2_transpose,
                       biases1,
                       biases2,
                       feature_size,
                       neurons_per_layer,
                       train_size,
                       epsilon,
                       batch_size,
                       num_threads,
                       hidden_layer_size);

    // Must perform a Cuda fence operation to ensure all threads have finished
    Kokkos::fence();

    backpropagate_end = std::chrono::high_resolution_clock::now();

    backpropagateTime += backpropagate_end - backpropagate_start;

    if (accuracy > max_accuracy) 
      max_accuracy = accuracy;
    if (loss < max_loss)
      max_loss = loss;

    // Output the results of the neural network
    fprintf(stdout, "|| Epoch: %d, Loss: %f, Accuracy: %f%\n", e, loss, accuracy);

    if (accuracy > accuracy_threshold) 
      e = num_epochs;
  }

  end = std::chrono::high_resolution_clock::now();

  //Store the time it took to run the function
  neuralNetworkTime += end - start;

  float accuracy = 0.0f;

  if (test_size > 0) {
    // Forward propogate the neural network to get the accuracy and loss for the test data
    ForwardPropogate(node_features,
                    forward_propogate_results_layer1,
                    forward_propogate_results_layer2,
                    weights1,
                    weights2,
                    biases1,
                    biases2,
                    feature_size,
                    neurons_per_layer,
                    start_test_index + test_size,
                    start_test_index,
                    hidden_layer_size);

    accuracy = CalculateAccuracy(forward_propogate_results_layer2, node_labels, start_test_index + test_size, start_test_index, neurons_per_layer);
  } else {
    accuracy = max_accuracy;
  }

  fprintf(stdout, "||========================================================RESULTS========================================================\n");
  
  auto totalTime = initialisationTime + generateNodeEmbeddingTime + neuralNetworkTime;

  fprintf(stdout, "\n||============================RESULTS============================\n");
  fprintf(stdout, "||Initialisation Time:                                %.6f   \n", initialisationTime.count());
  fprintf(stdout, "||Generate Node Embedding Time:                       %.6f   \n", generateNodeEmbeddingTime.count());
  fprintf(stdout, "||Neural Network Time:                                %.6f   \n", neuralNetworkTime.count());
  fprintf(stdout, "||Forward Propogate Time:                             %.6f   \n", forwardpropagateTime.count());
  fprintf(stdout, "||Back Propogate Time:                                %.6f   \n", backpropagateTime.count());
  fprintf(stdout, "||Utility Time:                                       %.6f   \n", utilityTime.count());
  fprintf(stdout, "||Total Time:                                         %.6f   \n", totalTime.count());
  fprintf(stdout, "||============================RESULTS============================\n");
  fprintf(stdout, "||Test Accuracy:                                      %.6f   \n", accuracy);
  fprintf(stdout, "||============================RESULTS============================\n");

  // Deallocate the memory as Kokkos was having deallocation occur after the Kokkos finalize function
  Kokkos::View<float**, MemLayout, MemSpace>::HostMirror deallocate_mirror_float;
  Kokkos::View<int*, MemSpace>::HostMirror deallocate_mirror_int;
  #ifdef KOKKOS_ENABLE_CUDA
    h_weights2 = deallocate_mirror_float;
    h_weights1 = deallocate_mirror_float;
    h_expected_results = deallocate_mirror_float;
    h_forward_propogate_results_layer2 = deallocate_mirror_float;
    h_forward_propogate_results_layer1 = deallocate_mirror_float;
    h_node_embeddings_layer_2 = deallocate_mirror_float;
    h_node_embeddings_layer_1 = deallocate_mirror_float;
    h_node_features = deallocate_mirror_float;
    h_node_labels = deallocate_mirror_int;
    h_gradients_layer2 = deallocate_mirror_float;
    h_gradients_layer1 = deallocate_mirror_float;
  #endif

  Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace> deallocate_float_2D_l;
  Kokkos::View<float**, Kokkos::LayoutRight, MemSpace> deallocate_float_2D_r;
  Kokkos::View<float*, MemSpace> deallocate_float_1D;
  Kokkos::View<int*, MemSpace> deallocate_int_1D;

  weights2 = deallocate_float_2D_r;
  weights1 = deallocate_float_2D_r;
  expected_results = deallocate_float_2D_r;
  forward_propogate_results_layer2 = deallocate_float_2D_r;
  forward_propogate_results_layer1 = deallocate_float_2D_r;
  node_embeddings_layer_2 = deallocate_float_2D_r;
  node_embeddings_layer_1 = deallocate_float_2D_r;
  node_features = deallocate_float_2D_r;
  node_labels = deallocate_int_1D;
  gradients_layer2 = deallocate_float_2D_r;
  gradients_layer1 = deallocate_float_2D_r;

  biases2 = deallocate_float_1D;
  biases1 = deallocate_float_1D;

  node_features_transpose = deallocate_float_2D_l;
  forward_propogate_results_layer1_transpose = deallocate_float_2D_l;
  gradients_layer1_transpose = deallocate_float_2D_l;
  gradients_layer2_transpose = deallocate_float_2D_l;

  graph_edges.clear();

  Kokkos::finalize();

  }
  
  return 0;
}
