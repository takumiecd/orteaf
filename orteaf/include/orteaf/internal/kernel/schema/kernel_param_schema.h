#pragma once

#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/param/param_key.h>
#include <orteaf/internal/kernel/param/param_list.h>
#include <orteaf/internal/kernel/param/param_transform.h>
#include <orteaf/kernel/param_id_tables.h>

#include <utility>

namespace orteaf::internal::kernel {

/**
 * @brief Helper to find a parameter in a ParamList.
 */
inline const Param *findParamInList(const ParamList &params, ParamId id) {
  return params.find(id);
}

inline const Param *findParamInList(const ParamList &params, ParamKey key) {
  return params.find(key);
}

/**
 * @brief Field type for kernel parameter schema.
 *
 * Associates a ParamId with its type and provides automatic extraction
 * from KernelArgs. Supports implicit conversion for convenient access.
 *
 * @tparam ID Parameter identifier
 * @tparam T Parameter value type
 *
 * Example:
 * @code
 * Field<ParamId::Alpha, float> alpha;
 * alpha.extract(args);
 * float value = alpha;  // Implicit conversion
 * @endcode
 */
template <ParamId ID, typename T> struct Field {
  static constexpr ParamId kId = ID;
  static constexpr ParamKey kKey = ParamKey::global(ID);
  using Type = T;
  T value{};

  /**
   * @brief Implicit conversion to value type.
   */
  constexpr operator T() const { return value; }

  /**
   * @brief Implicit conversion to value reference.
   */
  constexpr operator T &() { return value; }

  /**
   * @brief Explicit value accessor.
   */
  constexpr T &get() { return value; }

  /**
   * @brief Explicit value accessor (const).
   */
  constexpr const T &get() const { return value; }

  /**
   * @brief Extract value from parameter list.
   *
   * @param params Parameter list to extract from
   * @throws std::runtime_error if parameter not found or type mismatch
   */
  void extract(const ParamList &params) {
    const auto *param = findParamInList(params, kKey);
    if (!param) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Required parameter not found");
    }
    using ParamValueType =
        typename ::orteaf::generated::param_id_tables::ParamInfo<ID>::Type;
    const auto *val = param->template tryGet<ParamValueType>();
    if (!val) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Parameter type mismatch");
    }
    value = Transform<ParamValueType, T>(*val);
  }

  /**
   * @brief Extract value from kernel arguments.
   *
   * @tparam KernelArgs The kernel arguments type (KernelArgs)
   * @param args Kernel arguments containing parameters
   * @throws std::runtime_error if parameter not found or type mismatch
   */
  template <typename KernelArgs> void extract(const KernelArgs &args) {
    extract(args.paramList());
  }
};

/**
 * @brief Optional field type for kernel parameter schema.
 *
 * Similar to Field but allows missing parameters with default values.
 *
 * @tparam ID Parameter identifier
 * @tparam T Parameter value type
 */
template <ParamId ID, typename T> struct OptionalField {
  static constexpr ParamId kId = ID;
  static constexpr ParamKey kKey = ParamKey::global(ID);
  using Type = T;
  T value{};
  bool present = false;

  /**
   * @brief Construct with default value.
   */
  constexpr OptionalField() = default;

  /**
   * @brief Construct with explicit default value.
   */
  constexpr explicit OptionalField(T defaultValue) : value(defaultValue) {}

  /**
   * @brief Implicit conversion to value type.
   */
  constexpr operator T() const { return value; }

  /**
   * @brief Check if parameter was present.
   */
  explicit operator bool() const { return present; }

  /**
   * @brief Get value or default.
   */
  constexpr T valueOr(T defaultValue) const {
    return present ? value : defaultValue;
  }

  /**
   * @brief Extract value from parameter list (optional).
   *
   * Sets present flag if parameter found and type matches.
   *
   * @param params Parameter list to extract from
   */
  void extract(const ParamList &params) {
    const auto *param = findParamInList(params, kKey);
    if (!param) {
      present = false;
      return;
    }
    using ParamValueType =
        typename ::orteaf::generated::param_id_tables::ParamInfo<ID>::Type;
    const auto *val = param->template tryGet<ParamValueType>();
    if (!val) {
      present = false;
      return;
    }
    value = Transform<ParamValueType, T>(*val);
    present = true;
  }

  /**
   * @brief Extract value from kernel arguments (optional).
   *
   * Sets present flag if parameter found and type matches.
   *
   * @tparam KernelArgs The kernel arguments type (KernelArgs)
   * @param args Kernel arguments containing parameters
   */
  template <typename KernelArgs> void extract(const KernelArgs &args) {
    extract(args.paramList());
  }
};

/**
 * @brief Scoped field type for kernel parameter schema.
 *
 * Associates a ParamId with an operand-key scope and type.
 *
 * @tparam ID Parameter identifier
 * @tparam T Parameter value type
 * @tparam SID Operand identifier to scope to
 * @tparam Role Role (defaults to Data)
 */
template <ParamId ID, typename T, OperandId SID,
          Role Role = Role::Data>
struct ScopedField {
  static constexpr ParamId kId = ID;
  static constexpr OperandKey kOperandKey{SID, Role};
  static constexpr ParamKey kKey = ParamKey::scoped(ID, kOperandKey);
  using Type = T;
  T value{};

  /**
   * @brief Implicit conversion to value type.
   */
  constexpr operator T() const { return value; }

  /**
   * @brief Implicit conversion to value reference.
   */
  constexpr operator T &() { return value; }

  /**
   * @brief Explicit value accessor.
   */
  constexpr T &get() { return value; }

  /**
   * @brief Explicit value accessor (const).
   */
  constexpr const T &get() const { return value; }

  /**
   * @brief Extract value from parameter list.
   *
   * @param params Parameter list to extract from
   * @throws std::runtime_error if parameter not found or type mismatch
   */
  void extract(const ParamList &params) {
    const auto *param = findParamInList(params, kKey);
    if (!param) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Required parameter not found");
    }
    using ParamValueType =
        typename ::orteaf::generated::param_id_tables::ParamInfo<ID>::Type;
    const auto *val = param->template tryGet<ParamValueType>();
    if (!val) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Parameter type mismatch");
    }
    value = Transform<ParamValueType, T>(*val);
  }

  /**
   * @brief Extract value from kernel arguments.
   */
  template <typename KernelArgs> void extract(const KernelArgs &args) {
    extract(args.paramList());
  }
};

/**
 * @brief Optional scoped field type for kernel parameter schema.
 */
template <ParamId ID, typename T, OperandId SID,
          Role Role = Role::Data>
struct OptionalScopedField {
  static constexpr ParamId kId = ID;
  static constexpr OperandKey kOperandKey{SID, Role};
  static constexpr ParamKey kKey = ParamKey::scoped(ID, kOperandKey);
  using Type = T;
  T value{};
  bool present = false;

  constexpr OptionalScopedField() = default;
  constexpr explicit OptionalScopedField(T defaultValue) : value(defaultValue) {}

  constexpr operator T() const { return value; }
  explicit operator bool() const { return present; }

  constexpr T valueOr(T defaultValue) const {
    return present ? value : defaultValue;
  }

  void extract(const ParamList &params) {
    const auto *param = findParamInList(params, kKey);
    if (!param) {
      present = false;
      return;
    }
    using ParamValueType =
        typename ::orteaf::generated::param_id_tables::ParamInfo<ID>::Type;
    const auto *val = param->template tryGet<ParamValueType>();
    if (!val) {
      present = false;
      return;
    }
    value = Transform<ParamValueType, T>(*val);
    present = true;
  }

  template <typename KernelArgs> void extract(const KernelArgs &args) {
    extract(args.paramList());
  }
};

/**
 * @brief Base class for parameter schemas using CRTP.
 *
 * Provides static extract() method that calls extractAllFields()
 * on the derived schema class.
 *
 * @tparam Derived The derived schema class
 * @tparam KernelArgsType The kernel arguments type (defaults to auto-deduction)
 *
 * Example:
 * @code
 * struct MyParams : ParamSchema<MyParams> {
 *   Field<ParamId::Alpha, float> alpha;
 *   Field<ParamId::Beta, float> beta;
 *
 *   ORTEAF_EXTRACT_FIELDS(alpha, beta)
 * };
 *
 * auto params = MyParams::extract(args);
 * @endcode
 */
template <typename Derived> struct ParamSchema {
  /**
   * @brief Extract all fields from kernel arguments.
   *
   * Creates a new schema instance and calls extractAllFields()
   * on it to populate all field values.
   *
   * @tparam KernelArgs The kernel arguments type
   * @param args Kernel arguments containing parameters
   * @return Populated schema instance
   */
  template <typename KernelArgs>
  static Derived extract(const KernelArgs &args) {
    Derived schema;
    schema.extractAllFields(args);
    return schema;
  }
};

namespace detail {
// Helper for extracting multiple fields from ParamList
template <typename... Fields>
void extractFieldsFromList(const ParamList &params, Fields &...fields) {
  (fields.extract(params), ...);
}

// Helper for extracting multiple fields from KernelArgs
template <typename KernelArgs, typename... Fields>
void extractFields(const KernelArgs &args, Fields &...fields) {
  extractFieldsFromList(args.paramList(), fields...);
}

} // namespace detail

} // namespace orteaf::internal::kernel

/**
 * @brief Macro to generate extractAllFields() implementation.
 *
 * Automatically generates the extraction logic for all listed fields.
 * Each field's extract() method is called in sequence during extraction.
 *
 * Note: For binding parameters to encoder, use MpsKernelSession::bindParamsAt()
 * with explicit indices to ensure type safety with Metal shader bindings.
 *
 * Usage:
 * @code
 * struct MyParams : ParamSchema<MyParams> {
 *   Field<ParamId::Alpha, float> alpha;
 *   Field<ParamId::Beta, float> beta;
 *   Field<ParamId::Dim, std::size_t> dim;
 *
 *   ORTEAF_EXTRACT_FIELDS(alpha, beta, dim)
 * };
 *
 * auto params = MyParams::extract(args);
 * MpsKernelSession::bindParamsAt(encoder,
 *                                MpsKernelSession::Indices<0, 1, 2>{},
 *                                params.alpha, params.beta, params.dim);
 * @endcode
 */
#define ORTEAF_EXTRACT_FIELDS(...)                                             \
  template <typename KernelArgs>                                               \
  void extractAllFields(const KernelArgs &args) {                              \
    ::orteaf::internal::kernel::detail::extractFields(args, __VA_ARGS__);      \
  }
