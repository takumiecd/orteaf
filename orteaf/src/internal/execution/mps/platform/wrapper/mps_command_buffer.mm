/**
 * @file mps_command_buffer.mm
 * @brief Implementation of MPS/Metal command buffer helpers.
 */
#ifndef __OBJC__
#error                                                                         \
    "mps_command_buffer.mm must be compiled with an Objective-C++ compiler (__OBJC__ not defined)"
#endif
#include "orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_autorelease_pool.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_command_queue.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_event.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_objc_bridge.h"

#include "orteaf/internal/diagnostics/error/error.h"
#import <Metal/Metal.h>

namespace orteaf::internal::execution::mps::platform::wrapper {

using orteaf::internal::execution::mps::platform::wrapper::AutoreleasePool;

/**
 * @copydoc orteaf::internal::execution::mps::createCommandBuffer
 */
MpsCommandBuffer_t createCommandBuffer(MpsCommandQueue_t command_queue) {
  if (command_queue == nullptr) {
    using namespace orteaf::internal::diagnostics::error;
    throwError(OrteafErrc::NullPointer,
               "createCommandBuffer: command_queue cannot be nullptr");
  }
  AutoreleasePool pool{};
  id<MTLCommandQueue> objc_command_queue =
      objcFromOpaqueNoown<id<MTLCommandQueue>>(command_queue);
  id<MTLCommandBuffer> objc_command_buffer = [objc_command_queue commandBuffer];
  return (MpsCommandBuffer_t)opaqueFromObjcRetained(objc_command_buffer);
}

/**
 * @copydoc orteaf::internal::execution::mps::destroyCommandBuffer
 */
void destroyCommandBuffer(MpsCommandBuffer_t command_buffer) {
  if (command_buffer == nullptr)
    return;
  AutoreleasePool pool{};
  opaqueReleaseRetained(command_buffer);
}

/**
 * @copydoc orteaf::internal::execution::mps::encodeSignalEvent
 */
void encodeSignalEvent(MpsCommandBuffer_t command_buffer, MpsEvent_t event,
                       uint32_t value) {
  if (command_buffer == nullptr) {
    using namespace orteaf::internal::diagnostics::error;
    throwError(OrteafErrc::NullPointer,
               "encodeSignalEvent: command_buffer cannot be nullptr");
  }
  if (event == nullptr) {
    using namespace orteaf::internal::diagnostics::error;
    throwError(OrteafErrc::NullPointer,
               "encodeSignalEvent: event cannot be nullptr");
  }
  id<MTLCommandBuffer> objc_command_buffer =
      objcFromOpaqueNoown<id<MTLCommandBuffer>>(command_buffer);
  id<MTLSharedEvent> objc_event =
      objcFromOpaqueNoown<id<MTLSharedEvent>>(event);
  [objc_command_buffer encodeSignalEvent:objc_event value:value];
}

/**
 * @copydoc orteaf::internal::execution::mps::encodeWait
 */
void encodeWait(MpsCommandBuffer_t command_buffer, MpsEvent_t event,
                uint32_t value) {
  if (command_buffer == nullptr) {
    using namespace orteaf::internal::diagnostics::error;
    throwError(OrteafErrc::NullPointer,
               "encodeWait: command_buffer cannot be nullptr");
  }
  if (event == nullptr) {
    using namespace orteaf::internal::diagnostics::error;
    throwError(OrteafErrc::NullPointer, "encodeWait: event cannot be nullptr");
  }
  id<MTLCommandBuffer> objc_command_buffer =
      objcFromOpaqueNoown<id<MTLCommandBuffer>>(command_buffer);
  id<MTLSharedEvent> objc_event =
      objcFromOpaqueNoown<id<MTLSharedEvent>>(event);
  [objc_command_buffer encodeWaitForEvent:objc_event value:value];
}

/**
 * @copydoc orteaf::internal::execution::mps::commit
 */
void commit(MpsCommandBuffer_t command_buffer) {
  if (command_buffer == nullptr) {
    using namespace orteaf::internal::diagnostics::error;
    throwError(OrteafErrc::NullPointer,
               "commit: command_buffer cannot be nullptr");
  }
  id<MTLCommandBuffer> objc_command_buffer =
      objcFromOpaqueNoown<id<MTLCommandBuffer>>(command_buffer);
  [objc_command_buffer commit];
}

/**
 * @copydoc orteaf::internal::execution::mps::isCompleted
 */
bool isCompleted(MpsCommandBuffer_t command_buffer) {
  if (command_buffer == nullptr) {
    using namespace orteaf::internal::diagnostics::error;
    throwError(OrteafErrc::NullPointer,
               "isCompleted: command_buffer cannot be nullptr");
  }
  id<MTLCommandBuffer> objc_command_buffer =
      objcFromOpaqueNoown<id<MTLCommandBuffer>>(command_buffer);
  return objc_command_buffer.status == MTLCommandBufferStatusCompleted;
}

/**
 * @copydoc orteaf::internal::execution::mps::waitUntilCompleted
 */
void waitUntilCompleted(MpsCommandBuffer_t command_buffer) {
  if (command_buffer == nullptr) {
    using namespace orteaf::internal::diagnostics::error;
    throwError(OrteafErrc::NullPointer,
               "waitUntilCompleted: command_buffer cannot be nullptr");
  }
  id<MTLCommandBuffer> objc_command_buffer =
      objcFromOpaqueNoown<id<MTLCommandBuffer>>(command_buffer);
  [objc_command_buffer waitUntilCompleted];
}

/**
 * @copydoc orteaf::internal::execution::mps::getGPUStartTime
 */
double getGPUStartTime(MpsCommandBuffer_t command_buffer) {
  if (command_buffer == nullptr) {
    return 0.0;
  }
  id<MTLCommandBuffer> objc_command_buffer =
      objcFromOpaqueNoown<id<MTLCommandBuffer>>(command_buffer);
  return objc_command_buffer.GPUStartTime;
}

/**
 * @copydoc orteaf::internal::execution::mps::getGPUEndTime
 */
double getGPUEndTime(MpsCommandBuffer_t command_buffer) {
  if (command_buffer == nullptr) {
    return 0.0;
  }
  id<MTLCommandBuffer> objc_command_buffer =
      objcFromOpaqueNoown<id<MTLCommandBuffer>>(command_buffer);
  return objc_command_buffer.GPUEndTime;
}

/**
 * @copydoc orteaf::internal::execution::mps::getGPUDuration
 */
double getGPUDuration(MpsCommandBuffer_t command_buffer) {
  if (command_buffer == nullptr) {
    return 0.0;
  }
  id<MTLCommandBuffer> objc_command_buffer =
      objcFromOpaqueNoown<id<MTLCommandBuffer>>(command_buffer);
  const double start = objc_command_buffer.GPUStartTime;
  const double end = objc_command_buffer.GPUEndTime;
  if (start > 0.0 && end > 0.0) {
    return end - start;
  }
  return 0.0;
}

} // namespace orteaf::internal::execution::mps::platform::wrapper
