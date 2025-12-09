#pragma once

#include <atomic>
#include <type_traits>
#include <utility>

namespace orteaf::internal::base {

template <class HandleT, class ResourceT, class ManagerT> 
class SharedLease;

template <class ResourceT, class ManagerT> 
class SharedLeaseControlBlock {
public:
    using ResourceType = ResourceT;
    using ManagerType = ManagerT;
    // copy and move constructor is deleted because this is must be unique
    SharedLeaseControlBlock() = default;
    SharedLeaseControlBlock(ManagerType *manager, ResourceType resource)
        : manager_(manager), resource_(resource) {}
    SharedLeaseControlBlock(const SharedLeaseControlBlock&) = delete;
    SharedLeaseControlBlock& operator=(const SharedLeaseControlBlock&) = delete;
    SharedLeaseControlBlock(SharedLeaseControlBlock&&) = delete;
    SharedLeaseControlBlock& operator=(SharedLeaseControlBlock&&) = delete;

    void incrementRefCount() { ref_count_.fetch_add(1, std::memory_order_relaxed); }
    void decrementRefCount() { 
        if (ref_count_ == 1) {
            release();
            return;
        }
        ref_count_.fetch_sub(1, std::memory_order_relaxed);
    }

    decltype(auto) operator*() noexcept {
        if constexpr (std::is_pointer_v<ResourceT>) {
            return *resource_;
        } else {
            return (resource_);
        }
    }
    decltype(auto) operator*() const noexcept {
        if constexpr (std::is_pointer_v<ResourceT>) {
            return *resource_;
        } else {
            return (resource_);
        }
    }

    // Pointer access helper (returns raw pointer regardless of ResourceT being pointer or object).
    auto pointer() noexcept {
        if constexpr (std::is_pointer_v<ResourceT>) {
            return resource_;
        } else {
            return std::addressof(resource_);
        }
    }
    auto pointer() const noexcept {
        if constexpr (std::is_pointer_v<ResourceT>) {
            return resource_;
        } else {
            return std::addressof(resource_);
        }
    }

    template <class F>
    decltype(auto) with_resource(F&& f) {
        return static_cast<F&&>(f)(resource_);
    }

    template <class F>
    decltype(auto) with_resource(F&& f) const {
        return static_cast<F&&>(f)(resource_);
    }

private:
    void release() {
        if (manager_) {
            manager_->release(resource_);
            ref_count_.store(0, std::memory_order_relaxed);
            manager_ = nullptr;
            resource_ = nullptr;
        }
    }

    std::atomic<std::uint32_t> ref_count_{0};   
    ManagerType *manager_{nullptr};
    ResourceType resource_{nullptr};
};

/**
 * @brief A generic shared lease implementation that manages a resource using
 * reference counting provided by the ManagerT.
 *
 * This class is copyable and movable. Copying increments the reference count
 * via `manager_->retain(handle_)`. Destruction or release decrements the
 * reference count via `manager_->release(handle_)`.
 *
 * @tparam HandleT The type of the handle used to identify the resource in the
 * manager.
 * @tparam ResourceT The type of the actual resource being managed.
 * @tparam ManagerT The type of the manager that owns the resource. Must provide
 * `retain(HandleT)` and `release(HandleT)`.
 */
template <class HandleT, class ResourceT, class ManagerT> 
class SharedLease {
public:
  using HandleType = HandleT;
  using ResourceType = ResourceT;
  using ManagerType = ManagerT;
  using ControlBlockType = SharedLeaseControlBlock<ResourceType, ManagerType>;

  SharedLease() = default;
  SharedLease(const SharedLease &other) { copyFrom(other); }

  SharedLease &operator=(const SharedLease &other) {
    if (this != &other) {
      release();
      copyFrom(other);
    }
    return *this;
  }

  SharedLease(SharedLease &&other) noexcept
      : handle_(std::exchange(other.handle_, HandleType{})),
        control_block_(std::exchange(other.control_block_, nullptr)) {}

  SharedLease &operator=(SharedLease &&other) noexcept {
    if (this != &other) {
      release();
      handle_ = std::exchange(other.handle_, HandleType{});
      control_block_ = std::exchange(other.control_block_, nullptr);
    }
    return *this;
  }

  ~SharedLease() { release(); }

  auto operator->() { return control_block_->pointer(); }
  auto operator*() { return *control_block_; }

  auto operator->() const { return control_block_->pointer(); }
  auto operator*() const { return *control_block_; }

  auto pointer() { return control_block_->pointer(); }
  auto pointer() const { return control_block_->pointer(); }
  
  template <class F>
  void with_resource(F&& f) { control_block_->with_resource(f); }

  template <class F>
  void with_resource(F&& f) const { control_block_->with_resource(f); }


  explicit operator bool() const { return control_block_ != nullptr; }

  HandleType handle() const { return handle_; }

  void release() {
    if (control_block_) {
      control_block_->decrementRefCount();
      control_block_ = nullptr;
    }
  }

private:
  void copyFrom(const SharedLease &other) {
    if (other.control_block_) {
      control_block_ = other.control_block_;
      handle_ = other.handle_;
      control_block_->incrementRefCount();
    } else {
      control_block_ = nullptr;
    }
  }

  HandleType handle_{};
  ControlBlockType *control_block_{nullptr};
};

} // namespace orteaf::internal::base
