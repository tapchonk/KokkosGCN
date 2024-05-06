#ifndef GENERATENODEEMBEDDINGS_H
#define GENERATENODEEMBEDDINGS_H
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

#define KokkosGraphType Kokkos::StaticCrsGraph<int, Kokkos::LayoutRight, MemSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>, int>
#define KokkosNodeType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>
#define KokkosNodeEmbeddingType Kokkos::View<float**, Kokkos::LayoutRight, MemSpace>

int GenerateNodeEmbeddingLayer(KokkosGraphType graph,
                               KokkosNodeType node_features,
                               KokkosNodeEmbeddingType node_embeddings_layer_1,
                               long long start_index,
                               long long feature_size,
                               long long node_embed);



#endif