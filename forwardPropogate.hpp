#ifndef FORWARDPROPOGATE_H
#define FORWARDPROPOGATE_H
#include <Kokkos_Core.hpp>
#include <Kokkos_StaticCrsGraph.hpp>

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
#define MemSpace Kokkos::CudaSpace
#endif

#if !defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
#define MemSpace Kokkos::OpenMP
#endif

#ifdef KOKKOS_ENABLE_SERIAL
#define MemSpace Kokkos::Serial
#endif

#define KokkosNodeEmbeddingType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define ForwardPropogateResultsType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define WeightTypeDevice Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define BiasTypeDevice Kokkos::View<float*, MemSpace>

int ForwardPropogate(KokkosNodeEmbeddingType node_embeddings_layer_1,
                     ForwardPropogateResultsType forward_propogate_results_layer1,
                     ForwardPropogateResultsType forward_propogate_results_layer2,
                     WeightTypeDevice weights1,
                     WeightTypeDevice weights2,
                     BiasTypeDevice biases1,
                     BiasTypeDevice biases2,
                     int feature_size,
                     int neurons_per_layer,
                     int num_vertices,
                     int start_index,
                     int hidden_layer_size);



#endif