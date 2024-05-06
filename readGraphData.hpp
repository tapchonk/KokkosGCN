#ifndef READGRAPHDATA_H
#define READGRAPHDATA_H
#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
#define MemSpace Kokkos::CudaSpace
#define NodeLabelsType Kokkos::View<int*, MemSpace>::HostMirror
#define NodeFeaturesType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>::HostMirror
#endif

#if !defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
#define MemSpace Kokkos::OpenMP
#define NodeLabelsType Kokkos::View<int*, MemSpace>
#define NodeFeaturesType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#endif

#ifndef MemSpace
#define MemSpace Kokkos::Serial
#define NodeLabelsType Kokkos::View<int*, MemSpace>
#define NodeFeaturesType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#endif

int readGraphData( std::string label_file,
                   std::string feature_file,
                   std::string edge_file,
                   NodeLabelsType node_labels,
                   NodeFeaturesType node_features,
                   std::vector<std::vector<int>> *graph_edges,
                   long long num_vertices,
                   long long num_edges,
                   long long feature_size);

#endif