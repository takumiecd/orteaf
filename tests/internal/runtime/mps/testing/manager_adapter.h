#pragma once

#include <type_traits>
#include <utility>

#include "tests/internal/runtime/mps/testing/backend_mock_expectations.h"

namespace orteaf::tests::runtime::mps::testing {

template <class ManagerT, class Provider>
class ManagerAdapter {
public:
    using Manager = ManagerT;
    using Context = typename Provider::Context;

    void bind(Manager& manager, Context& context) {
        manager_ = &manager;
        context_ = &context;
    }

    Manager& manager() {
        return *manager_;
    }

    const Manager& manager() const {
        return *manager_;
    }

    template <class... Args>
    void expectCreateCommandQueues(Args&&... args) {
        if constexpr (Provider::is_mock) {
            auto& mock = Provider::mock(*context_);
            BackendMockExpectations::expectCreateCommandQueues(mock, std::forward<Args>(args)...);
        } else {
            (void)sizeof...(args);
        }
    }

    template <class... Args>
    void expectCreateEvents(Args&&... args) {
        if constexpr (Provider::is_mock) {
            auto& mock = Provider::mock(*context_);
            BackendMockExpectations::expectCreateEvents(mock, std::forward<Args>(args)...);
        } else {
            (void)sizeof...(args);
        }
    }

    template <class... Args>
    void expectDestroyCommandQueues(Args&&... args) {
        if constexpr (Provider::is_mock) {
            auto& mock = Provider::mock(*context_);
            BackendMockExpectations::expectDestroyCommandQueues(mock, std::forward<Args>(args)...);
        } else {
            (void)sizeof...(args);
        }
    }

    template <class... Args>
    void expectDestroyEvents(Args&&... args) {
        if constexpr (Provider::is_mock) {
            auto& mock = Provider::mock(*context_);
            BackendMockExpectations::expectDestroyEvents(mock, std::forward<Args>(args)...);
        } else {
            (void)sizeof...(args);
        }
    }

    template <class... Args>
    void expectDestroyCommandQueuesInOrder(Args&&... args) {
        if constexpr (Provider::is_mock) {
            auto& mock = Provider::mock(*context_);
            BackendMockExpectations::expectDestroyCommandQueuesInOrder(mock, std::forward<Args>(args)...);
        } else {
            (void)sizeof...(args);
        }
    }

    template <class... Args>
    void expectDestroyEventsInOrder(Args&&... args) {
        if constexpr (Provider::is_mock) {
            auto& mock = Provider::mock(*context_);
            BackendMockExpectations::expectDestroyEventsInOrder(mock, std::forward<Args>(args)...);
        } else {
            (void)sizeof...(args);
        }
    }

private:
    Manager* manager_{nullptr};
    Context* context_{nullptr};
};

}  // namespace orteaf::tests::runtime::mps::testing
