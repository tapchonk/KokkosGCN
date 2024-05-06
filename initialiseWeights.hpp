#ifndef INITIALISEWEIGHTS_H
#define INITIALISEWEIGHTS_H
#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
#define MemSpace Kokkos::CudaSpace
#define WeightType1 Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>::HostMirror
#define BiasType1 Kokkos::View<float*, MemSpace>::HostMirror
#define WeightUpdateType1 Kokkos::View<float***, Kokkos::LayoutRight, MemSpace>::HostMirror
#define ExpectedType1 Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>::HostMirror
#define NodeLabelsType Kokkos::View<int*, MemSpace>::HostMirror
#define BiasUpdateType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>::HostMirror
#endif

#if !defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
#define MemSpace Kokkos::OpenMP
#define WeightType1 Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define BiasType1 Kokkos::View<float*, MemSpace>
#define WeightUpdateType1 Kokkos::View<float***, Kokkos::LayoutRight, MemSpace>
#define ExpectedType1 Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define NodeLabelsType Kokkos::View<int*, MemSpace>
#define BiasUpdateType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#endif

#ifdef KOKKOS_ENABLE_SERIAL
#define MemSpace Kokkos::Serial
#define WeightType1 Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define BiasType1 Kokkos::View<float*, MemSpace>
#define WeightUpdateType1 Kokkos::View<float***, Kokkos::LayoutRight, MemSpace>
#define ExpectedType1 Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define NodeLabelsType Kokkos::View<int*, MemSpace>
#define BiasUpdateType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#endif

int InitialiseWeights(WeightType1 weights1,
                      WeightType1 weights2,
                      int feature_size,
                      int neurons_per_layer,
                      int hidden_layer_size,
                      int num_threads   );

int InitialiseExpected (ExpectedType1 expected_results,
                        NodeLabelsType node_labels,
                        long long num_vertices,
                        int neurons_per_layer);

#endif