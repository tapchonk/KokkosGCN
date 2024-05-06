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

#include "generateNodeEmbeddings.hpp"

/**
 * @brief Generate the node embedding by utilising a mean aggregation of the node features and the node neighbourhood
 * We can stack the operations for each layer of the graph, if we for example wanted to extend the convolutions over
 * a two neighbourhood layer, we would just need to perform this function twice and so on. 
 * 
/**
 * @param graph: The graph representation containing the connectivity information between nodes.
 * @param node_features: The matrix containing the features of each node.
 * @param node_embeddings_layer_1: The matrix to store the generated node embeddings for the first layer.
 * @param num_vertices: The total number of vertices in the graph.
 * @param feature_size: The size of the feature vector for each node.
 * @param node_embed: The number of nodes to generate embeddings for.
 * @return int: Returns 1 upon successful generation of node embeddings.
 */
int GenerateNodeEmbeddingLayer(KokkosGraphType graph,
                               KokkosNodeType node_features,
                               KokkosNodeEmbeddingType node_embeddings_layer_1,
                               long long start_index,
                               long long feature_size,
                               long long node_embed) {

  // Generate the node embedding by utilising a mean aggregation
  // Each layer of neighbourhoods will be aggregated into the node embedding with the average of that nodes layer out degree
  // We also include the node features from the starting node in the node embedding
  using current_range_policy = Kokkos::RangePolicy<MemSpace::execution_space, Kokkos::Schedule<Kokkos::Dynamic>>;

  using Kokkos::parallel_for;
  using Kokkos::parallel_reduce;
  using Kokkos::TeamPolicy;
  using Kokkos::TeamThreadRange;
  using Kokkos::ThreadVectorRange;
  using Kokkos::TeamVectorRange;
  using Kokkos::sqrt;
  using ExecutionSpace = MemSpace::execution_space;

  
  parallel_for(TeamPolicy<ExecutionSpace>(node_embed, Kokkos::AUTO),
  KOKKOS_LAMBDA (const TeamPolicy<ExecutionSpace>::member_type& team1) {
    int i = team1.league_rank();
    float normalised_length_1 = Kokkos::sqrt((float)(graph.rowConst(i).length + 1));
    parallel_for(TeamVectorRange(team1, feature_size), [=] (int j) {
      node_features(i, j) = node_features(i, j) / normalised_length_1;
    });
  });

  #if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)

  parallel_for(TeamPolicy<ExecutionSpace>(node_embed, Kokkos::AUTO),
    KOKKOS_LAMBDA (const TeamPolicy<ExecutionSpace>::member_type& team1) {
      int i = team1.league_rank();
      auto row_view_level_1 = graph.rowConst(i);
      int length_1 = row_view_level_1.length + 1;
      if (length_1 == 1) {
        // if the node has no neighbours, just use the node features
        parallel_for(TeamThreadRange(team1, feature_size),
          [=] (int j) {
            node_embeddings_layer_1(i, j) = node_features(i, j);
        } );
        return;
      }
      float length_1_div = (float)length_1;

      // for each of the neighbours:
      parallel_for(TeamThreadRange(team1, feature_size),
        [=] (int k) {
          float sum = 0.0f;
        parallel_reduce(ThreadVectorRange(team1, length_1 - 1),
          [=] (int j, float& temp_sum) {
            temp_sum += node_features(row_view_level_1(j), k);
        }, sum);
        node_embeddings_layer_1(i, k) = sum;
        // also take into account the source node features
        node_embeddings_layer_1(i, k) += node_features(i, k);
        // all nodes have equal weights
        // Divide the node embedding by the number of nodes in the neighbourhood
        node_embeddings_layer_1(i, k) /= sqrt((float)length_1_div);
      });

      return;

  });

  #endif

  #if (!defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)) || defined(KOKKOS_ENABLE_SERIAL)
  /*
  parallel_for(TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>(node_embed, 1),
    KOKKOS_LAMBDA (const TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>::member_type& team1) {
      int i = team1.league_rank();
      auto row_view_level_1 = graph.rowConst(i);
      int length_1 = row_view_level_1.length + 1;
      float length_1_div = (float)length_1;

      if (length_1 == 0) {
        // if the node has no neighbours, just use the node features
        parallel_for(TeamThreadRange(team1, feature_size), [=] (int m) {
          node_embeddings_layer_1(i, m) = node_features(i, m);
        });
        return;
      }

      // for each of the neighbours: 
      parallel_for(TeamThreadRange(team1, feature_size), [=] (int j) {
        float sum = 0.0f;
        parallel_reduce(ThreadVectorRange(team1, length_1 - 1), [=] (int m, float& temp_sum) {
          temp_sum += node_features(row_view_level_1(m), j);
        }, sum);
        node_embeddings_layer_1(i, j) = sum;
      });
      // also take into account the source node features
      parallel_for(TeamThreadRange(team1, feature_size), [=] (int m) {
        node_embeddings_layer_1(i, m) += node_features(i, m);
        // all nodes have equal weights
      });
      // Divide the node embedding by the number of nodes in the neighbourhood
      parallel_for(TeamThreadRange(team1, feature_size), [=] (int m) {
        node_embeddings_layer_1(i, m) /= sqrt(length_1_div);
      });
      return;
  })*/

  parallel_for( "generate_node_embedding",  current_range_policy(start_index, node_embed), KOKKOS_LAMBDA (long long i) {
    auto row_view_level_1 = graph.rowConst(i);
    int length_1 = row_view_level_1.length + 1;
    float length_1_div = (float)length_1;

    // initialise node embedding to 0
    for (int m = 0; m < feature_size; m++) {
      node_embeddings_layer_1(i, m) = 0.0f;
    }

    // for each of the neighbours: 
    for (int j = 0; j < length_1 - 1; ++j) {
      for (int m = 0; m < feature_size; m++) {
        node_embeddings_layer_1(i, m) += node_features(row_view_level_1(j), m);
      }
    }
    // also take into account the source node features
    for (int m = 0; m < feature_size; m++) {
      node_embeddings_layer_1(i, m) += node_features(i, m);
      // split weighting 10/90 between node features and neighbourhood
    }
    // Divide the node embedding by the number of nodes in the neighbourhood
    for (int m = 0; m < feature_size; m++) {
      node_embeddings_layer_1(i, m) /= sqrt(length_1_div);
    }
  });

  #endif

  return 1;
}

/*

    parallel_for(TeamPolicy<MemSpace::execution_space>(node_embed, Kokkos::AUTO),
      KOKKOS_LAMBDA (const TeamPolicy<MemSpace::execution_space>::member_type& team1) {
        int i = team1.league_rank();
        auto row_view_level_1 = graph.rowConst(i);
        int length_1 = row_view_level_1.length + 1;
        float length_1_div = (float)length_1;

        if (length_1 == 0) {
          // if the node has no neighbours, just use the node features
          parallel_for(TeamThreadRange(team1, feature_size), [=] (int m) {
            node_embeddings_layer_1(i, m) = node_features(i, m);
          });
          return;
        }

        // for each of the neighbours: 
        parallel_for(TeamThreadRange(team1, length_1 - 1), [&] (int j) {
          parallel_for(ThreadVectorRange(team1, feature_size), [=] (int m) {
            node_embeddings_layer_1(i, m) += node_features(row_view_level_1(j), m);
          });
        });

        // also take into account the source node features
        parallel_for(TeamThreadRange(team1, feature_size), [=] (int m) {
          node_embeddings_layer_1(i, m) += node_features(i, m);
          // all nodes have equal weights
        });
        
        // Divide the node embedding by the number of nodes in the neighbourhood
        parallel_for(TeamThreadRange(team1, feature_size), [=] (int m) {
          node_embeddings_layer_1(i, m) /= length_1_div;
        });
        return;
    });

*/