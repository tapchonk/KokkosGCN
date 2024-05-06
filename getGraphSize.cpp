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

#include "getGraphSize.hpp"

/**
 * @brief Get the graph size from the input files
 * This utilises a simple file reading scheme to extract the number of vertices, edges, and the feature size from the input files.
 * 
 * @param vertex_file The path to the file containing the vertices information.
 * @param edge_file The path to the file containing the edges information.
 * @param feature_file The path to the file containing the features information.
 * @param num_vertices Pointer to a variable to store the number of vertices.
 * @param num_edges Pointer to a variable to store the number of edges.
 * @param feature_size Pointer to a variable to store the feature size.
 * @return int Returns 0 on success, -1 on failure.
 */
int getGraphSize(std::string vertex_file,
                 std::string edge_file,
                 std::string feature_file,
                 long long *num_vertices,
                 long long *num_edges,
                 long long *feature_size) {

  //Read in the graph size data from the files
  FILE *file = fopen(vertex_file.c_str(), "r");
  if (file == NULL) {
      printf("Failed to open the input file.\n");
      return -1;
  }

  // Read input values into their declared variables 
  fscanf(file, "%ld\n", num_vertices);
  fclose(file);

  file = fopen(edge_file.c_str(), "r");
  if (file == NULL) {
      printf("Failed to open the input file.\n");
      return -1;
  }

  // Read input values into their declared variables 
  fscanf(file, "%ld\n", num_edges);
  fclose(file);

  file = fopen(feature_file.c_str(), "r");
  char c;
  if (file == NULL) {
      printf("Failed to open the input file.\n");
      return -1;
  }

  long long counter = 0;

  // Determine maximal feature length
  while ((c = fgetc(file)) != EOF) {
    if (c == ',') {
      counter++;
    }
    if (c == '\n') {
      break;
    }
  }
  counter++;
  fclose(file);

  *feature_size = counter;

  return 0;
}
