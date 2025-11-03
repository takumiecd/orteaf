#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace orteaf::internal::error {

/**
 * 共通の例外スロー関数。内部実装から利用し、ユーザー層には伝播させずに変換する。
 * 実装は最小限にしておき、必要に応じてロギングやトレースを追加する。
 */
[[noreturn]] inline void throwRuntimeError(std::string_view message) {
    throw std::runtime_error(std::string(message));
}

/**
 * 例外変換のラッパー。呼び出し側で使用することで例外種別を統一できる。
 */
template <typename Fn>
auto wrapAndRethrow(Fn&& fn) -> decltype(auto) {
    try {
        return fn();
    } catch (const std::exception&) {
        throw;
    }
}

}  // namespace orteaf::internal::error
