/**
 * @file cuda_graph.cu
 * @brief Implementation of CUDA Graph helpers (create/destroy/capture/instantiate/launch).
 */
#include "orteaf/internal/backend/cuda/cuda_graph.h"
#include "orteaf/internal/backend/cuda/cuda_check.h"
#include "orteaf/internal/backend/cuda/cuda_objc_bridge.h"

#ifdef ORTEAF_ENABLE_CUDA
#include <cuda.h>
#include "orteaf/internal/diagnostics/error/error_impl.h"
#endif

namespace orteaf::internal::backend::cuda {

/**
 * @copydoc orteaf::internal::backend::cuda::create_graph
 */
CUgraph_t create_graph() {
#ifdef ORTEAF_ENABLE_CUDA
    CUgraph graph;
    CU_CHECK(cuGraphCreate(&graph, 0));
    return opaque_from_objc_noown<CUgraph_t, CUgraph>(graph);
#else
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::create_graph_exec
 */
CUgraphExec_t create_graph_exec(CUgraph_t graph) {
#ifdef ORTEAF_ENABLE_CUDA
    if (graph == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "create_graph_exec: graph cannot be nullptr");
    }
    CUgraph objc_graph = objc_from_opaque_noown<CUgraph>(graph);
    CUgraphExec graph_exec;
    CU_CHECK(cuGraphInstantiateWithFlags(&graph_exec, objc_graph, 0));
    return opaque_from_objc_noown<CUgraphExec_t, CUgraphExec>(graph_exec);
#else
    (void)graph;
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::destroy_graph
 */
void destroy_graph(CUgraph_t graph) {
#ifdef ORTEAF_ENABLE_CUDA
    CUgraph objc_graph = objc_from_opaque_noown<CUgraph>(graph);
    CU_CHECK(cuGraphDestroy(objc_graph));
#else
    (void)graph;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::destroy_graph_exec
 */
void destroy_graph_exec(CUgraphExec_t graph_exec) {
#ifdef ORTEAF_ENABLE_CUDA
    CUgraphExec objc_graph_exec = objc_from_opaque_noown<CUgraphExec>(graph_exec);
    CU_CHECK(cuGraphExecDestroy(objc_graph_exec));
#else
    (void)graph_exec;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::begin_graph_capture
 */
void begin_graph_capture(CUstream_t stream) {
#ifdef ORTEAF_ENABLE_CUDA
    if (stream == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "begin_graph_capture: stream cannot be nullptr");
    }
    CUstream objc_stream = objc_from_opaque_noown<CUstream>(stream);
    CU_CHECK(cuStreamBeginCapture(objc_stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
#else
    (void)stream;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::end_graph_capture
 */
void end_graph_capture(CUstream_t stream, CUgraph_t* graph) {
#ifdef ORTEAF_ENABLE_CUDA
    if (stream == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "end_graph_capture: stream cannot be nullptr");
    }
    if (graph == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "end_graph_capture: graph cannot be nullptr");
    }
    CUstream objc_stream = objc_from_opaque_noown<CUstream>(stream);
    CUgraph objc_graph;
    CU_CHECK(cuStreamEndCapture(objc_stream, &objc_graph));
    *graph = opaque_from_objc_noown<CUgraph_t, CUgraph>(objc_graph);
#else
    (void)stream;
    (void)graph;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::instantiate_graph
 */
void instantiate_graph(CUgraph_t graph, CUgraphExec_t* graph_exec) {
#ifdef ORTEAF_ENABLE_CUDA
    if (graph == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "instantiate_graph: graph cannot be nullptr");
    }
    if (graph_exec == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "instantiate_graph: graph_exec cannot be nullptr");
    }
    CUgraph objc_graph = objc_from_opaque_noown<CUgraph>(graph);
    CUgraphExec objc_graph_exec;
    CU_CHECK(cuGraphInstantiateWithFlags(&objc_graph_exec, objc_graph, 0));
    *graph_exec = opaque_from_objc_noown<CUgraphExec_t, CUgraphExec>(objc_graph_exec);
#else
    (void)graph;
    (void)graph_exec;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::graph_launch
 */
void graph_launch(CUgraphExec_t graph_exec, CUstream_t stream) {
#ifdef ORTEAF_ENABLE_CUDA
    if (graph_exec == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "graph_launch: graph_exec cannot be nullptr");
    }
    if (stream == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "graph_launch: stream cannot be nullptr");
    }
    CUgraphExec objc_graph_exec = objc_from_opaque_noown<CUgraphExec>(graph_exec);
    CUstream objc_stream = objc_from_opaque_noown<CUstream>(stream);
    CU_CHECK(cuGraphLaunch(objc_graph_exec, objc_stream));
#else
    (void)graph_exec;
    (void)stream;
#endif
}

} // namespace orteaf::internal::backend::cuda
