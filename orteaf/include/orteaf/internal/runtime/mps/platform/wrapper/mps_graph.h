#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>

#include "orteaf/internal/runtime/mps/platform/wrapper/mps_buffer.h"
#include "orteaf/internal/runtime/mps/platform/wrapper/mps_command_queue.h"
#include "orteaf/internal/runtime/mps/platform/wrapper/mps_device.h"

namespace orteaf::internal::runtime::mps::platform::wrapper {

struct MPSGraph_st;
using MPSGraph_t = MPSGraph_st*;

struct MPSGraphTensor_st;
using MPSGraphTensor_t = MPSGraphTensor_st*;

struct MPSGraphOperation_st;
using MPSGraphOperation_t = MPSGraphOperation_st*;

struct MPSGraphExecutable_st;
using MPSGraphExecutable_t = MPSGraphExecutable_st*;

struct MPSGraphTensorData_st;
using MPSGraphTensorData_t = MPSGraphTensorData_st*;

static_assert(sizeof(MPSGraph_t) == sizeof(void*), "MPSGraph must be pointer-sized.");
static_assert(sizeof(MPSGraphTensor_t) == sizeof(void*), "MPSGraphTensor must be pointer-sized.");
static_assert(sizeof(MPSGraphOperation_t) == sizeof(void*), "MPSGraphOperation must be pointer-sized.");
static_assert(sizeof(MPSGraphExecutable_t) == sizeof(void*), "MPSGraphExecutable must be pointer-sized.");
static_assert(sizeof(MPSGraphTensorData_t) == sizeof(void*), "MPSGraphTensorData must be pointer-sized.");

enum class MpsGraphDataType : std::uint32_t {
  kInvalid = 0,
  kFloat16,
  kFloat32,
  kInt32,
  kInt64,
  kBool,
};

struct MpsGraphFeed {
  MPSGraphTensor_t tensor{nullptr};
  MPSGraphTensorData_t data{nullptr};
};

MPSGraph_t createGraph();

void destroyGraph(MPSGraph_t graph);

MPSGraphTensorData_t
createGraphTensorDataFromBuffer(MPSBuffer_t buffer, const std::int64_t* shape,
                                std::size_t shape_rank,
                                MpsGraphDataType data_type);

void destroyGraphTensorData(MPSGraphTensorData_t data);

MPSGraphExecutable_t compileGraph(
    MPSGraph_t graph, MPSDevice_t device, const MpsGraphFeed* feeds,
    std::size_t feed_count, const MPSGraphTensor_t* target_tensors,
    std::size_t target_tensor_count,
    const MPSGraphOperation_t* target_operations,
    std::size_t target_operation_count);

std::size_t runGraphExecutable(
    MPSGraphExecutable_t executable, MPSCommandQueue_t command_queue,
    const MpsGraphFeed* feeds, std::size_t feed_count,
    const MPSGraphTensor_t* target_tensors, std::size_t target_tensor_count,
    const MPSGraphOperation_t* target_operations,
    std::size_t target_operation_count,
    MPSGraphTensorData_t* out_tensor_data, std::size_t out_capacity);

void destroyGraphExecutable(MPSGraphExecutable_t executable);

}  // namespace orteaf::internal::runtime::mps::platform::wrapper

#endif  // ORTEAF_ENABLE_MPS
