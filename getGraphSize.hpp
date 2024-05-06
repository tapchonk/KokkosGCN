#ifndef GETGRAPHSIZE_H
#define GETGRAPHSIZE_H
#include <Kokkos_Core.hpp>

int getGraphSize(std::string vertex_file,
                 std::string edge_file,
                 std::string feature_file,
                 long long *num_vertices,
                 long long *num_edges,
                 long long *feature_size);

#endif