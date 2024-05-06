#ifndef ACCURACYERRORUTIL_H
#define ACCURACYERRORUTIL_H
#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
#define MemSpace Kokkos::CudaSpace
#endif

#if !defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
#define MemSpace Kokkos::OpenMP
#endif

#ifdef KOKKOS_ENABLE_SERIAL
#define MemSpace Kokkos::Serial
#endif

#define ForwardPropogateResultsType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define ExpectedResultsType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define NodeLabelsTypeDevice Kokkos::View<int*, MemSpace>

float MeanSquareError(ForwardPropogateResultsType forward_propogate_results_layer2,
                       ExpectedResultsType expected_results,
                       int num_vertices,
                       int neurons_per_layer);

float CalculateAccuracy(ForwardPropogateResultsType forward_propogate_results_layer2,
                         NodeLabelsTypeDevice node_labels,
                         int num_vertices,
                         int start_index,
                         int neurons_per_layer);

#endif