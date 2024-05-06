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

#include "accuracyErrorUtil.hpp"

/**
 * @brief Calculate the mean square error loss.
 * 
 * @param forward_propogate_results_layer2 The results of the forward propagation in layer 2.
 * @param expected_results The expected results for each vertex.
 * @param num_vertices The total number of vertices.
 * @param neurons_per_layer The number of neurons per layer.
 * @return float The mean square error loss.
 */
float MeanSquareError(ForwardPropogateResultsType forward_propogate_results_layer2,
                       ExpectedResultsType expected_results,
                       int num_vertices,
                       int neurons_per_layer) {

  // Calculate the mean square error loss
  using current_range_policy = Kokkos::RangePolicy<MemSpace::execution_space, Kokkos::Schedule<Kokkos::Static>>;
  float loss = 0.0f;

  Kokkos::fence();

  Kokkos::parallel_reduce(Kokkos::TeamPolicy<MemSpace::execution_space>(num_vertices, Kokkos::AUTO),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<MemSpace::execution_space>::member_type& team, float& update) {
      int i = team.league_rank();
      float loss = 0.0f;
      Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team, neurons_per_layer), [=](int j, float& temp_loss) {
        float diff = expected_results(i, j) - forward_propogate_results_layer2(i, j);
        temp_loss += diff * diff;
      }, loss);

      //Kokkos::single(Kokkos::PerTeam(team), [&] () {
      if (team.team_rank() == 0)
        update += loss;
      //});
  }, loss );

  Kokkos::fence();

  loss /= (float)num_vertices;
  loss *= 0.5f;

  return loss;
}

/**
 * @brief CalculateAccuracy calculates the accuracy of the neural network.
 * 
 * @param forward_propogate_results_layer2 The results of the forward propagation in layer 2.
 * @param node_labels The labels of the nodes.
 * @param num_vertices The total number of vertices.
 * @param start_index The starting index for calculation.
 * @param neurons_per_layer The number of neurons per layer.
 * @return float The accuracy of the neural network.
 */
float CalculateAccuracy(ForwardPropogateResultsType forward_propogate_results_layer2,
                         NodeLabelsTypeDevice node_labels,
                         int num_vertices,
                         int start_index,
                         int neurons_per_layer) {

  // Calculate the accuracy of the neural network
  using current_range_policy = Kokkos::RangePolicy<MemSpace::execution_space, Kokkos::Schedule<Kokkos::Static>>;
  int total = num_vertices - start_index;
  int correct = 0;
  typedef Kokkos::MaxLoc<float,int>::value_type max_loc_type;
  
  Kokkos::fence();

  Kokkos::parallel_reduce(Kokkos::TeamPolicy<MemSpace::execution_space>(total, Kokkos::AUTO),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<MemSpace::execution_space>::member_type& team, int& update) {
      max_loc_type max_loc;
      max_loc.val = 0.0f;
      max_loc.loc = 0;
      int i = team.league_rank() + start_index;
      Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team, neurons_per_layer), [=](int j, max_loc_type& local_max_loc) {
        if (forward_propogate_results_layer2(i, j) > local_max_loc.val) {
          local_max_loc.val = forward_propogate_results_layer2(i, j);
          local_max_loc.loc = j;
        }
      }, Kokkos::MaxLoc<float,int>(max_loc));

      //Kokkos::single(Kokkos::PerTeam(team), [&] () {
        if (node_labels(i) == max_loc.loc && team.team_rank() == 0) {
          update += 1;
        
        }
      //});
  }, correct );

  Kokkos::fence();

  return ((float)correct/(float)total)*100.0f;
}