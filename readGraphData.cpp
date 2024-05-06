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

#include "readGraphData.hpp"

/**
 * @brief Read the graph data from the input files, and store the data in the appropriate data structures
 * I was considering multithreading this function, but I think it would be better to keep it simple for now.
 * 
 * @param label_file: The path to the file containing the node labels.
 * @param feature_file: The path to the file containing the node features.
 * @param edge_file: The path to the file containing the graph edges.
 * @param node_labels: The data structure to store the node labels.
 * @param node_features: The data structure to store the node features.
 * @param graph_edges: A pointer to a vector of vectors to store the graph edges.
 * @param num_vertices: The number of vertices in the graph.
 * @param num_edges: The number of edges in the graph.
 * @param feature_size: The size of each node feature vector.
 * @return int: Returns 0 if the graph data is successfully read, -1 otherwise.
 */
int readGraphData( std::string label_file,
                   std::string feature_file,
                   std::string edge_file,
                   NodeLabelsType node_labels,
                   NodeFeaturesType node_features,
                   std::vector<std::vector<int>> *graph_edges,
                   long long num_vertices,
                   long long num_edges,
                   long long feature_size) {

  //extract the node labels
  FILE *file = fopen(label_file.c_str(), "r");
  if (file == NULL) {
      printf("Failed to open the input file.\n");
      return -1;
  }
  for (int i = 0; i < num_vertices; i++) {
    int label;
    fscanf(file, "%d\n", &label);
    node_labels(i) = label;
  }
  fclose(file);
  //extract the node features
  file = fopen(feature_file.c_str(), "r");
  if (file == NULL) {
      printf("Failed to open the input file.\n");
      return -1;
  }
  for (int i = 0; i < num_vertices; i++) {
    for (int j = 0; j < feature_size; j++) {
      float feature;
      fscanf(file, "%f,", &feature);
      node_features(i, j) = feature;
    }
    fscanf(file, "\n");
  }
  fclose(file);

  //insert "num_vertices" number of empty vectors into the graph_edges vector
  //extract the edges
  file = fopen(edge_file.c_str(), "r");
  if (file == NULL) {
      printf("Failed to open the input file.\n");
      return -1;
  }
  for (int i = 0; i < num_edges; i++) {
    int src, dst;
    fscanf(file, "%d,%d\n", &src, &dst);
    graph_edges->at(src).push_back(dst);
    graph_edges->at(dst).push_back(src);
  }
  fclose(file);
  return 0;
}