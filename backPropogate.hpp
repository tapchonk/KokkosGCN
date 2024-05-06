#ifndef BACKPROPOGATE_H
#define BACKPROPOGATE_H
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

#define KokkosNodeEmbeddingTypeTranspose Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace>
#define ForwardPropogateResultsType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define ExpectedResultsType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define WeightTypeDevice Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define GradientsTypeDevice Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define BiasTypeDevice Kokkos::View<float*, MemSpace>
#define ForwardPropogateResultsTypeTranspose Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace>
#define GradientsTypeDeviceTranspose Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace>

void BackpropagateBatch(KokkosNodeEmbeddingTypeTranspose node_embeddings_layer1_transpose,
                        ForwardPropogateResultsType forward_propagate_results_layer1,
                        ForwardPropogateResultsType forward_propagate_results_layer2,
                        ForwardPropogateResultsTypeTranspose forward_prop_results_layer1_transpose,
                        ExpectedResultsType expected_results,
                        WeightTypeDevice weights1,
                        WeightTypeDevice weights2,
                        GradientsTypeDevice gradients_layer1,
                        GradientsTypeDevice gradients_layer2,
                        GradientsTypeDeviceTranspose gradients_layer1_transpose,
                        GradientsTypeDeviceTranspose gradients_layer2_transpose,
                        BiasTypeDevice biases1,
                        BiasTypeDevice biases2,
                        int feature_size,
                        int neurons_per_layer,
                        int num_vertices,
                        float learning_rate,
                        int batch_size,
                        int num_threads,
                        int hidden_layer_size);



#endif