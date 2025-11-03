#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace orteaf::internal::diagnostics::error {

/**
 * @brief orteaf 固有のエラーコード。
 *
 * 成功(0)、未知エラー、引数不正、状態不正、バックエンド不可、メモリ不足、
 * 一般的な操作失敗といった分類を保持する。
 */
enum class OrteafErrc {
    Success = 0,              ///< 成功
    Unknown = 1,              ///< 原因不明のエラー
    InvalidArgument = 2,      ///< 引数が不正
    InvalidState = 3,         ///< オブジェクト状態が不正
    BackendUnavailable = 4,   ///< バックエンドが利用不可
    OutOfMemory = 5,          ///< メモリ確保に失敗
    OperationFailed = 6,      ///< その他の操作失敗
};

/**
 * @brief ORTEAF 用の error_category。
 */
class OrteafErrorCategory : public std::error_category {
public:
    /// カテゴリ名。
    const char* name() const noexcept override;

    /// エラーコードに対応するメッセージ。
    std::string message(int condition) const override;
};

/// @brief エラーカテゴリのシングルトンを取得。
const std::error_category& orteaf_error_category();

/// @brief エラーコードを生成するヘルパ。
std::error_code make_error_code(OrteafErrc errc);

/**
 * @brief ORTEAF 専用の system_error 派生クラス。
 *
 * エラーコードとメッセージを保持し、`std::system_error` として扱える。
 */
class OrteafError : public std::system_error {
public:
    OrteafError(std::error_code ec, std::string message);
    OrteafError(OrteafErrc errc, std::string message);
    explicit OrteafError(std::error_code ec);

private:
    static std::string compose_message(const std::error_code& ec, std::string_view message);
};

/// @brief エラーを送出するヘルパ。OrteafErrc から OrteafError を投げる。
[[noreturn]] void throw_error(OrteafErrc errc, std::string message = {});

/// @brief std::error_code から OrteafError を投げる。
[[noreturn]] void throw_error(std::error_code ec, std::string message = {});

/**
 * @brief Result 型の内部実装。
 */
namespace detail {

template <typename T>
class OrteafResultImpl {
public:
    static OrteafResultImpl success(T value);
    static OrteafResultImpl failure(OrteafError error);

    bool has_value() const noexcept;
    bool has_error() const noexcept;

    T& value() &;
    const T& value() const&;
    T&& value() &&;

    T value_or(T default_value) const&;

    OrteafError error() const;

private:
    template <typename V>
    OrteafResultImpl(std::in_place_index_t<0>, V&& value);
    OrteafResultImpl(std::in_place_index_t<1>, OrteafError error);

    std::optional<T> value_;
    std::optional<OrteafError> error_;
};

template <>
class OrteafResultImpl<void> {
public:
    static OrteafResultImpl success();
    static OrteafResultImpl failure(OrteafError error);

    bool has_value() const noexcept;
    bool has_error() const noexcept;

    void value() const;
    void value_or() const;

    OrteafError error() const;

private:
    OrteafResultImpl();
    bool has_value_{true};
    std::optional<OrteafError> error_{};
};

}  // namespace detail

/**
 * @brief ORTEAF 向けの Result 型。
 */
template <typename T>
class OrteafResult {
public:
    static OrteafResult success(T value);
    static OrteafResult failure(OrteafErrc errc, std::string message = {});
    static OrteafResult failure(OrteafError error);

    template <typename... Args>
    static OrteafResult failure_with(OrteafErrc errc, Args&&... args);

    bool has_value() const noexcept;
    bool has_error() const noexcept;

    T& value() &;
    const T& value() const&;
    T&& value() &&;

    template <typename U>
    T value_or(U&& default_value) const&;

    OrteafError error() const;

private:
    explicit OrteafResult(detail::OrteafResultImpl<T>&& impl);

    detail::OrteafResultImpl<T> impl_;
};

template <>
class OrteafResult<void> {
public:
    static OrteafResult success();
    static OrteafResult failure(OrteafErrc errc, std::string message = {});
    static OrteafResult failure(OrteafError error);

    bool has_value() const noexcept;
    bool has_error() const noexcept;

    void value() const;
    void value_or() const;

    OrteafError error() const;

private:
    explicit OrteafResult(detail::OrteafResultImpl<void>&& impl);

    detail::OrteafResultImpl<void> impl_;
};

/**
 * @brief 関数を実行し、例外を Result に変換する。
 */
template <typename Fn>
auto capture_result(Fn&& fn) -> OrteafResult<std::invoke_result_t<Fn>>;

/**
 * @brief Result から値を取り出し、失敗している場合は例外を投げる。
 */
template <typename T>
T unwrap_or_throw(OrteafResult<T>&& result);

/// @copydoc unwrap_or_throw(OrteafResult<T>&&)
void unwrap_or_throw(OrteafResult<void>&& result);

/// C 関数ポインタ互換の capture_result。
OrteafResult<void> capture_result(void (*fn)());

}  // namespace orteaf::internal::diagnostics::error

namespace std {

// OrteafErrc を std::error_code として扱えるようにする。
template <>
struct is_error_code_enum<orteaf::internal::diagnostics::error::OrteafErrc> : true_type {};

}  // namespace std

#include "error_impl.h"
