/**
 * @file cuda_graph.h
 * @brief CUDA Graph creation, capture, instantiation, and launch helpers.
 *
 * Thin wrappers over CUDA Driver API for CUDA Graph workflows. When CUDA is
 * disabled, functions are available but behave as no-ops and return neutral
 * values (e.g., nullptr) where applicable.
 */
#pragma once

#include "orteaf/internal/backend/cuda/cuda_stream.h"

struct CUgraph_st;
using CUgraph_t = CUgraph_st*;

struct CUgraphExec_st;
using CUgraphExec_t = CUgraphExec_st*;

namespace orteaf::internal::backend::cuda {

/**
 * @brief Create an empty CUDA graph.
 * @return Opaque graph handle, or nullptr when CUDA is disabled.
 * @throws std::system_error On CUDA driver error (via `OrteafErrc`).
 */
CUgraph_t create_graph();

/**
 * @brief Create an executable instance from a graph.
 * @param graph Opaque graph handle
 * @return Opaque executable graph, or nullptr when CUDA is disabled.
 * @throws std::system_error On CUDA driver error (via `OrteafErrc`).
 */
CUgraphExec_t create_graph_exec(CUgraph_t graph);

/**
 * @brief Destroy a graph.
 * @param graph Opaque graph handle (ignored if nullptr)
 * @throws std::system_error On CUDA driver error (via `OrteafErrc`).
 */
void destroy_graph(CUgraph_t graph);

/**
 * @brief Destroy an executable graph instance.
 * @param graph_exec Opaque graph exec handle (ignored if nullptr)
 * @throws std::system_error On CUDA driver error (via `OrteafErrc`).
 */
void destroy_graph_exec(CUgraphExec_t graph_exec);

/**
 * @brief Begin capturing operations on a stream into a graph (global mode).
 * @param stream Opaque stream handle
 * @throws std::system_error On CUDA driver error (via `OrteafErrc`).
 */
void begin_graph_capture(CUstream_t stream);

/**
 * @brief End capture and return the captured graph.
 * @param stream Opaque stream handle
 * @param graph Out parameter for the captured opaque graph handle
 * @throws std::system_error On CUDA driver error (via `OrteafErrc`).
 */
void end_graph_capture(CUstream_t stream, CUgraph_t* graph);

/**
 * @brief Instantiate an executable graph from a graph.
 * @param graph Opaque graph handle
 * @param graph_exec Out parameter for opaque executable graph handle
 * @throws std::system_error On CUDA driver error (via `OrteafErrc`).
 */
void instantiate_graph(CUgraph_t graph, CUgraphExec_t* graph_exec);

/**
 * @brief Launch an executable graph on a stream.
 * @param graph_exec Opaque executable graph handle
 * @param stream Opaque stream handle
 * @throws std::system_error On CUDA driver error (via `OrteafErrc`).
 */
void graph_launch(CUgraphExec_t graph_exec, CUstream_t stream);

} // namespace orteaf::internal::backend::cuda