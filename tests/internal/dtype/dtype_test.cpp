#include "orteaf/internal/dtype/dtype.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>

namespace dtype = orteaf::internal;

// ============================================================================
// DTypeBasic - 基本機能テスト
// ============================================================================

TEST(DTypeBasic, EnumValuesAreDefined) {
    // 等価クラス: 各dtypeカテゴリの代表値でenum値が正しく定義されているか
    EXPECT_EQ(static_cast<std::uint16_t>(dtype::DType::Bool), 0u);
    EXPECT_EQ(static_cast<std::uint16_t>(dtype::DType::I8), 1u);
    EXPECT_EQ(static_cast<std::uint16_t>(dtype::DType::I32), 3u);
    EXPECT_EQ(static_cast<std::uint16_t>(dtype::DType::I64), 4u);
    EXPECT_EQ(static_cast<std::uint16_t>(dtype::DType::U8), 5u);
    EXPECT_EQ(static_cast<std::uint16_t>(dtype::DType::U32), 7u);
    EXPECT_EQ(static_cast<std::uint16_t>(dtype::DType::U64), 8u);
    EXPECT_EQ(static_cast<std::uint16_t>(dtype::DType::F8E4M3), 9u);
    EXPECT_EQ(static_cast<std::uint16_t>(dtype::DType::F8E5M2), 10u);
    EXPECT_EQ(static_cast<std::uint16_t>(dtype::DType::F16), 11u);
    EXPECT_EQ(static_cast<std::uint16_t>(dtype::DType::F32), 12u);
    EXPECT_EQ(static_cast<std::uint16_t>(dtype::DType::F64), 13u);
}

TEST(DTypeBasic, CountIsCorrect) {
    // 境界条件: DType::Countが正しい値（14）か
    EXPECT_EQ(static_cast<std::size_t>(dtype::DType::Count), 14u);
    EXPECT_EQ(dtype::kDTypeCount, 14u);
    EXPECT_EQ(dtype::kDTypeCount, static_cast<std::size_t>(dtype::DType::Count));
}

TEST(DTypeBasic, IndexConversion) {
    // 返し値アサート: ToIndexとFromIndexが正しく動作するか
    EXPECT_EQ(dtype::ToIndex(dtype::DType::Bool), 0u);
    EXPECT_EQ(dtype::ToIndex(dtype::DType::I8), 1u);
    EXPECT_EQ(dtype::ToIndex(dtype::DType::I32), 3u);
    EXPECT_EQ(dtype::ToIndex(dtype::DType::F64), 13u);

    EXPECT_EQ(dtype::FromIndex(0), dtype::DType::Bool);
    EXPECT_EQ(dtype::FromIndex(1), dtype::DType::I8);
    EXPECT_EQ(dtype::FromIndex(3), dtype::DType::I32);
    EXPECT_EQ(dtype::FromIndex(13), dtype::DType::F64);
}

TEST(DTypeBasic, IndexConversionRoundTrip) {
    // 動作の順序: ToIndexとFromIndexが双方向に一致するか
    for (std::size_t i = 0; i < dtype::kDTypeCount; ++i) {
        const auto dtype = dtype::FromIndex(i);
        EXPECT_EQ(dtype::ToIndex(dtype), i);
    }

    for (const auto dtype : dtype::kAllDTypes) {
        const auto index = dtype::ToIndex(dtype);
        EXPECT_EQ(dtype::FromIndex(index), dtype);
    }
}

TEST(DTypeBasic, IndexValidation) {
    // 境界条件: IsValidIndexが正しく動作するか
    EXPECT_TRUE(dtype::IsValidIndex(0));
    EXPECT_TRUE(dtype::IsValidIndex(dtype::kDTypeCount - 1));
    EXPECT_FALSE(dtype::IsValidIndex(dtype::kDTypeCount));
    EXPECT_FALSE(dtype::IsValidIndex(dtype::kDTypeCount + 1));
    EXPECT_FALSE(dtype::IsValidIndex(1000u));
}

TEST(DTypeBasic, AllDTypesArrayIsComplete) {
    // パラメーターテスト: kAllDTypes配列がすべてのdtypeを含むか
    EXPECT_EQ(dtype::kAllDTypes.size(), dtype::kDTypeCount);
    for (std::size_t i = 0; i < dtype::kDTypeCount; ++i) {
        EXPECT_EQ(dtype::ToIndex(dtype::kAllDTypes[i]), i);
    }
}

// ============================================================================
// DTypeMetadata - メタデータ取得テスト
// ============================================================================

TEST(DTypeMetadata, IdOf) {
    // 返し値アサート: IdOfが正しいIDを返すか
    EXPECT_EQ(dtype::IdOf(dtype::DType::Bool), std::string_view("Bool"));
    EXPECT_EQ(dtype::IdOf(dtype::DType::I32), std::string_view("I32"));
    EXPECT_EQ(dtype::IdOf(dtype::DType::F64), std::string_view("F64"));
    EXPECT_EQ(dtype::IdOf(dtype::DType::U8), std::string_view("U8"));
    EXPECT_EQ(dtype::IdOf(dtype::DType::F8E4M3), std::string_view("F8E4M3"));
    EXPECT_EQ(dtype::IdOf(dtype::DType::F16), std::string_view("F16"));
}

TEST(DTypeMetadata, DisplayNameOf) {
    // 返し値アサート: DisplayNameOfが正しい表示名を返すか
    EXPECT_EQ(dtype::DisplayNameOf(dtype::DType::Bool), std::string_view("bool"));
    EXPECT_EQ(dtype::DisplayNameOf(dtype::DType::I32), std::string_view("int32"));
    EXPECT_EQ(dtype::DisplayNameOf(dtype::DType::F64), std::string_view("float64"));
    EXPECT_EQ(dtype::DisplayNameOf(dtype::DType::U8), std::string_view("uint8"));
    EXPECT_EQ(dtype::DisplayNameOf(dtype::DType::F8E4M3), std::string_view("float8 (e4m3)"));
    EXPECT_EQ(dtype::DisplayNameOf(dtype::DType::F8E5M2), std::string_view("float8 (e5m2)"));
    EXPECT_EQ(dtype::DisplayNameOf(dtype::DType::F16), std::string_view("float16"));
}

TEST(DTypeMetadata, CategoryOf) {
    // 等価クラス: カテゴリごとに代表値をテスト
    EXPECT_EQ(dtype::CategoryOf(dtype::DType::Bool), std::string_view("boolean"));
    EXPECT_EQ(dtype::CategoryOf(dtype::DType::I8), std::string_view("signed_integer"));
    EXPECT_EQ(dtype::CategoryOf(dtype::DType::I32), std::string_view("signed_integer"));
    EXPECT_EQ(dtype::CategoryOf(dtype::DType::U8), std::string_view("unsigned_integer"));
    EXPECT_EQ(dtype::CategoryOf(dtype::DType::U32), std::string_view("unsigned_integer"));
    EXPECT_EQ(dtype::CategoryOf(dtype::DType::F8E4M3), std::string_view("floating_point"));
    EXPECT_EQ(dtype::CategoryOf(dtype::DType::F8E5M2), std::string_view("floating_point"));
    EXPECT_EQ(dtype::CategoryOf(dtype::DType::F32), std::string_view("floating_point"));
    EXPECT_EQ(dtype::CategoryOf(dtype::DType::F64), std::string_view("floating_point"));
}

TEST(DTypeMetadata, AllTypesHaveMetadata) {
    // パラメーターテスト: すべてのdtypeでメタデータが正しいか
    for (const auto dtype : dtype::kAllDTypes) {
        const auto id = dtype::IdOf(dtype);
        const auto display_name = dtype::DisplayNameOf(dtype);
        const auto category = dtype::CategoryOf(dtype);

        EXPECT_FALSE(id.empty());
        EXPECT_FALSE(display_name.empty());
        EXPECT_FALSE(category.empty());
    }
}

// ============================================================================
// DTypeProperties - 型プロパティテスト
// ============================================================================

TEST(DTypeProperties, SizeOf) {
    // 返し値アサート: SizeOfが正しいサイズを返すか
    EXPECT_EQ(dtype::SizeOf(dtype::DType::Bool), sizeof(bool));
    EXPECT_EQ(dtype::SizeOf(dtype::DType::I8), sizeof(std::int8_t));
    EXPECT_EQ(dtype::SizeOf(dtype::DType::I32), sizeof(std::int32_t));
    EXPECT_EQ(dtype::SizeOf(dtype::DType::I64), sizeof(std::int64_t));
    EXPECT_EQ(dtype::SizeOf(dtype::DType::U8), sizeof(std::uint8_t));
    EXPECT_EQ(dtype::SizeOf(dtype::DType::F8E4M3), sizeof(::orteaf::internal::Float8E4M3));
    EXPECT_EQ(dtype::SizeOf(dtype::DType::F8E5M2), sizeof(::orteaf::internal::Float8E5M2));
    EXPECT_EQ(dtype::SizeOf(dtype::DType::F32), sizeof(float));
    EXPECT_EQ(dtype::SizeOf(dtype::DType::F64), sizeof(double));
}

TEST(DTypeProperties, AlignmentOf) {
    // 返し値アサート: AlignmentOfが正しいアライメントを返すか
    EXPECT_EQ(dtype::AlignmentOf(dtype::DType::Bool), alignof(bool));
    EXPECT_EQ(dtype::AlignmentOf(dtype::DType::I8), alignof(std::int8_t));
    EXPECT_EQ(dtype::AlignmentOf(dtype::DType::I32), alignof(std::int32_t));
    EXPECT_EQ(dtype::AlignmentOf(dtype::DType::I64), alignof(std::int64_t));
    EXPECT_EQ(dtype::AlignmentOf(dtype::DType::U8), alignof(std::uint8_t));
    EXPECT_EQ(dtype::AlignmentOf(dtype::DType::F8E4M3), alignof(::orteaf::internal::Float8E4M3));
    EXPECT_EQ(dtype::AlignmentOf(dtype::DType::F8E5M2), alignof(::orteaf::internal::Float8E5M2));
    EXPECT_EQ(dtype::AlignmentOf(dtype::DType::F32), alignof(float));
    EXPECT_EQ(dtype::AlignmentOf(dtype::DType::F64), alignof(double));
}

TEST(DTypeProperties, PromotionPriority) {
    // 等価クラス: 優先度の順序が正しいか
    EXPECT_EQ(dtype::PromotionPriority(dtype::DType::Bool), 10);
    EXPECT_EQ(dtype::PromotionPriority(dtype::DType::I8), 100);
    EXPECT_EQ(dtype::PromotionPriority(dtype::DType::I16), 200);
    EXPECT_EQ(dtype::PromotionPriority(dtype::DType::I32), 300);
    EXPECT_EQ(dtype::PromotionPriority(dtype::DType::I64), 400);
    EXPECT_EQ(dtype::PromotionPriority(dtype::DType::U8), 110);
    EXPECT_EQ(dtype::PromotionPriority(dtype::DType::U32), 310);
    EXPECT_EQ(dtype::PromotionPriority(dtype::DType::F8E4M3), 420);
    EXPECT_EQ(dtype::PromotionPriority(dtype::DType::F8E5M2), 430);
    EXPECT_EQ(dtype::PromotionPriority(dtype::DType::F16), 500);
    EXPECT_EQ(dtype::PromotionPriority(dtype::DType::F32), 600);
    EXPECT_EQ(dtype::PromotionPriority(dtype::DType::F64), 700);
}

TEST(DTypeProperties, PromotionPriorityOrdering) {
    // 動作の順序: 優先度が昇順になっているか
    EXPECT_LT(dtype::PromotionPriority(dtype::DType::Bool),
              dtype::PromotionPriority(dtype::DType::I8));
    EXPECT_LT(dtype::PromotionPriority(dtype::DType::I8),
              dtype::PromotionPriority(dtype::DType::I32));
    EXPECT_LT(dtype::PromotionPriority(dtype::DType::I32),
              dtype::PromotionPriority(dtype::DType::F8E4M3));
    EXPECT_LT(dtype::PromotionPriority(dtype::DType::F8E4M3),
              dtype::PromotionPriority(dtype::DType::F8E5M2));
    EXPECT_LT(dtype::PromotionPriority(dtype::DType::F8E5M2),
              dtype::PromotionPriority(dtype::DType::F16));
    EXPECT_LT(dtype::PromotionPriority(dtype::DType::I32),
              dtype::PromotionPriority(dtype::DType::F32));
    EXPECT_LT(dtype::PromotionPriority(dtype::DType::F32),
              dtype::PromotionPriority(dtype::DType::F64));
}

TEST(DTypeProperties, AllTypesHaveProperties) {
    // パラメーターテスト: すべてのdtypeでプロパティが正しいか
    for (const auto dtype : dtype::kAllDTypes) {
        const auto size = dtype::SizeOf(dtype);
        const auto alignment = dtype::AlignmentOf(dtype);
        const auto priority = dtype::PromotionPriority(dtype);

        EXPECT_GT(size, 0u);
        EXPECT_GT(alignment, 0u);
        EXPECT_GE(priority, 0);
    }
}

// ============================================================================
// DTypePromotion - 型プロモーション機能テスト
// ============================================================================

TEST(DTypePromotion, SameType) {
    // 条件テスト: 同じ型同士のプロモーション
    EXPECT_EQ(dtype::Promote(dtype::DType::I32, dtype::DType::I32), dtype::DType::I32);
    EXPECT_EQ(dtype::Promote(dtype::DType::F32, dtype::DType::F32), dtype::DType::F32);
    EXPECT_EQ(dtype::Promote(dtype::DType::Bool, dtype::DType::Bool), dtype::DType::Bool);
}

TEST(DTypePromotion, DifferentPriorities) {
    // 条件テスト: 異なる優先度の型同士
    EXPECT_EQ(dtype::Promote(dtype::DType::I8, dtype::DType::I32), dtype::DType::I32);
    EXPECT_EQ(dtype::Promote(dtype::DType::I32, dtype::DType::I8), dtype::DType::I32);
    EXPECT_EQ(dtype::Promote(dtype::DType::I32, dtype::DType::F32), dtype::DType::F32);
    EXPECT_EQ(dtype::Promote(dtype::DType::F32, dtype::DType::I32), dtype::DType::F32);
}

TEST(DTypePromotion, BooleanOverrides) {
    // 等価クラス: boolean × boolean のオーバーライド
    EXPECT_EQ(dtype::Promote(dtype::DType::Bool, dtype::DType::Bool), dtype::DType::Bool);
    // boolean × float のオーバーライド
    EXPECT_EQ(dtype::Promote(dtype::DType::Bool, dtype::DType::F16), dtype::DType::F16);
    EXPECT_EQ(dtype::Promote(dtype::DType::Bool, dtype::DType::F32), dtype::DType::F32);
    EXPECT_EQ(dtype::Promote(dtype::DType::Bool, dtype::DType::F64), dtype::DType::F64);
    EXPECT_EQ(dtype::Promote(dtype::DType::F16, dtype::DType::Bool), dtype::DType::F16);
    EXPECT_EQ(dtype::Promote(dtype::DType::F32, dtype::DType::Bool), dtype::DType::F32);
    EXPECT_EQ(dtype::Promote(dtype::DType::F64, dtype::DType::Bool), dtype::DType::F64);
}

TEST(DTypePromotion, SignedInteger) {
    // 等価クラス: signed × signed
    EXPECT_EQ(dtype::Promote(dtype::DType::I8, dtype::DType::I16), dtype::DType::I16);
    EXPECT_EQ(dtype::Promote(dtype::DType::I8, dtype::DType::I32), dtype::DType::I32);
    EXPECT_EQ(dtype::Promote(dtype::DType::I16, dtype::DType::I32), dtype::DType::I32);
    EXPECT_EQ(dtype::Promote(dtype::DType::I32, dtype::DType::I64), dtype::DType::I64);
}

TEST(DTypePromotion, UnsignedInteger) {
    // 等価クラス: unsigned × unsigned
    EXPECT_EQ(dtype::Promote(dtype::DType::U8, dtype::DType::U16), dtype::DType::U16);
    EXPECT_EQ(dtype::Promote(dtype::DType::U8, dtype::DType::U32), dtype::DType::U32);
    EXPECT_EQ(dtype::Promote(dtype::DType::U16, dtype::DType::U32), dtype::DType::U32);
    EXPECT_EQ(dtype::Promote(dtype::DType::U32, dtype::DType::U64), dtype::DType::U64);
}

TEST(DTypePromotion, SignedUnsigned) {
    // 等価クラス: signed × unsigned
    EXPECT_EQ(dtype::Promote(dtype::DType::I32, dtype::DType::U32), dtype::DType::U32);
    EXPECT_EQ(dtype::Promote(dtype::DType::U32, dtype::DType::I32), dtype::DType::U32);
    EXPECT_EQ(dtype::Promote(dtype::DType::I64, dtype::DType::U64), dtype::DType::U64);
    EXPECT_EQ(dtype::Promote(dtype::DType::U64, dtype::DType::I64), dtype::DType::U64);
}

TEST(DTypePromotion, IntegerFloat) {
    // 等価クラス: integer × float
    EXPECT_EQ(dtype::Promote(dtype::DType::I32, dtype::DType::F32), dtype::DType::F32);
    EXPECT_EQ(dtype::Promote(dtype::DType::F32, dtype::DType::I32), dtype::DType::F32);
    EXPECT_EQ(dtype::Promote(dtype::DType::I64, dtype::DType::F64), dtype::DType::F64);
    EXPECT_EQ(dtype::Promote(dtype::DType::F64, dtype::DType::I64), dtype::DType::F64);
}

TEST(DTypePromotion, Symmetry) {
    // 動作の順序: プロモーションの対称性
    for (const auto lhs : dtype::kAllDTypes) {
        for (const auto rhs : dtype::kAllDTypes) {
            EXPECT_EQ(dtype::Promote(lhs, rhs), dtype::Promote(rhs, lhs));
        }
    }
}

TEST(DTypePromotion, ResultIsValid) {
    // パラメーターテスト: すべてのdtype組み合わせでプロモーションをテスト
    for (const auto lhs : dtype::kAllDTypes) {
        for (const auto rhs : dtype::kAllDTypes) {
            const auto result = dtype::Promote(lhs, rhs);
            const auto result_index = dtype::ToIndex(result);
            EXPECT_LT(result_index, dtype::kDTypeCount);
        }
    }
}

TEST(DTypePromotion, ResultPriorityIsAtLeastInputs) {
    // パラメーターテスト: 結果の優先度が両方の入力の優先度以上であることを確認
    for (const auto lhs : dtype::kAllDTypes) {
        for (const auto rhs : dtype::kAllDTypes) {
            const auto result = dtype::Promote(lhs, rhs);
            const auto lhs_priority = dtype::PromotionPriority(lhs);
            const auto rhs_priority = dtype::PromotionPriority(rhs);
            const auto result_priority = dtype::PromotionPriority(result);

            EXPECT_GE(result_priority, lhs_priority);
            EXPECT_GE(result_priority, rhs_priority);
        }
    }
}

TEST(DTypePromotion, BoundaryConditions) {
    // 境界条件: 最小優先度と最大優先度の組み合わせ
    EXPECT_EQ(dtype::Promote(dtype::DType::Bool, dtype::DType::F64), dtype::DType::F64);
    EXPECT_EQ(dtype::Promote(dtype::DType::F64, dtype::DType::Bool), dtype::DType::F64);
}

// ============================================================================
// DTypeCasting - 型キャスト機能テスト
// ============================================================================

TEST(DTypeCasting, SameTypeImplicit) {
    // 等価クラス: 同じ型へのキャスト
    EXPECT_TRUE(dtype::CanImplicitlyCast(dtype::DType::I32, dtype::DType::I32));
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::F32, dtype::DType::F32));
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::I32, dtype::DType::I32));
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::F32, dtype::DType::F32));
}

TEST(DTypeCasting, NarrowToWideImplicit) {
    // 等価クラス: 小さい型から大きい型への暗黙的キャスト
    EXPECT_TRUE(dtype::CanImplicitlyCast(dtype::DType::I8, dtype::DType::I32));
    EXPECT_TRUE(dtype::CanImplicitlyCast(dtype::DType::I16, dtype::DType::I32));
    EXPECT_TRUE(dtype::CanImplicitlyCast(dtype::DType::I32, dtype::DType::I64));
    EXPECT_TRUE(dtype::CanImplicitlyCast(dtype::DType::F32, dtype::DType::F64));
}

TEST(DTypeCasting, WideToNarrowExplicit) {
    // 等価クラス: 大きい型から小さい型への明示的キャスト
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::I32, dtype::DType::I8));
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::I64, dtype::DType::I32));
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::F64, dtype::DType::F32));
}

TEST(DTypeCasting, IntegerToFloatImplicit) {
    // 等価クラス: 整数から浮動小数点への暗黙的キャスト
    EXPECT_TRUE(dtype::CanImplicitlyCast(dtype::DType::I32, dtype::DType::F32));
    EXPECT_TRUE(dtype::CanImplicitlyCast(dtype::DType::I64, dtype::DType::F32));
    EXPECT_TRUE(dtype::CanImplicitlyCast(dtype::DType::I32, dtype::DType::F64));
    EXPECT_TRUE(dtype::CanImplicitlyCast(dtype::DType::I64, dtype::DType::F64));
}

TEST(DTypeCasting, FloatToIntegerExplicit) {
    // 等価クラス: 浮動小数点から整数への明示的キャスト
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::F32, dtype::DType::I32));
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::F64, dtype::DType::I32));
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::F32, dtype::DType::I64));
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::F64, dtype::DType::I64));
}

TEST(DTypeCasting, BooleanCasting) {
    // 条件テスト: Bool型のキャスト
    // Bool -> I8, U8 は暗黙的キャスト可能
    EXPECT_TRUE(dtype::CanImplicitlyCast(dtype::DType::Bool, dtype::DType::I8));
    EXPECT_TRUE(dtype::CanImplicitlyCast(dtype::DType::Bool, dtype::DType::U8));
    // Bool -> I32, F32 などは明示的キャストのみ
    EXPECT_FALSE(dtype::CanImplicitlyCast(dtype::DType::Bool, dtype::DType::I32));
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::Bool, dtype::DType::I32));
    EXPECT_FALSE(dtype::CanImplicitlyCast(dtype::DType::Bool, dtype::DType::F32));
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::Bool, dtype::DType::F32));
}

TEST(DTypeCasting, ImplicitImpliesExplicit) {
    // 動作の順序: 暗黙的キャスト可能な場合、明示的キャストも可能であることを確認
    for (const auto from : dtype::kAllDTypes) {
        for (const auto to : dtype::kAllDTypes) {
            if (dtype::CanImplicitlyCast(from, to)) {
                EXPECT_TRUE(dtype::CanExplicitlyCast(from, to))
                    << "Implicit cast from " << dtype::IdOf(from) << " to " << dtype::IdOf(to)
                    << " should imply explicit cast";
            }
        }
    }
}

TEST(DTypeCasting, CastModeConsistency) {
    // 条件テスト: CanCastがCastModeで正しく動作するか
    for (const auto from : dtype::kAllDTypes) {
        for (const auto to : dtype::kAllDTypes) {
            EXPECT_EQ(dtype::CanCast(from, to, dtype::CastMode::Implicit),
                      dtype::CanImplicitlyCast(from, to));
            EXPECT_EQ(dtype::CanCast(from, to, dtype::CastMode::Explicit),
                      dtype::CanExplicitlyCast(from, to));
        }
    }
}

TEST(DTypeCasting, NegativeCases) {
    // ネガティブテスト: 定義されていないキャストがfalseを返すか
    // F64 -> I32 は暗黙的キャスト不可（明示的のみ）
    EXPECT_FALSE(dtype::CanImplicitlyCast(dtype::DType::F64, dtype::DType::I32));
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::F64, dtype::DType::I32));
    // F64 -> F32 は暗黙的キャスト不可（明示的のみ）
    EXPECT_FALSE(dtype::CanImplicitlyCast(dtype::DType::F64, dtype::DType::F32));
    EXPECT_TRUE(dtype::CanExplicitlyCast(dtype::DType::F64, dtype::DType::F32));
}

// ============================================================================
// DTypeComputeType - 計算用型の取得テスト
// ============================================================================

TEST(DTypeComputeType, Basic) {
    // 返し値アサート: ComputeTypeが正しい計算用型を返すか
    EXPECT_EQ(dtype::ComputeType(dtype::DType::Bool), dtype::DType::Bool);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::I8), dtype::DType::I32);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::I32), dtype::DType::I32);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::I64), dtype::DType::I64);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::F8E4M3), dtype::DType::F16);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::F8E5M2), dtype::DType::F16);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::F16), dtype::DType::F32);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::F32), dtype::DType::F32);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::F64), dtype::DType::F64);
}

TEST(DTypeComputeType, CategoryRepresentatives) {
    // 等価クラス: カテゴリごとの代表値
    // boolean
    EXPECT_EQ(dtype::ComputeType(dtype::DType::Bool), dtype::DType::Bool);
    // signed_integer
    EXPECT_EQ(dtype::ComputeType(dtype::DType::I8), dtype::DType::I32);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::I16), dtype::DType::I32);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::I32), dtype::DType::I32);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::I64), dtype::DType::I64);
    // unsigned_integer
    EXPECT_EQ(dtype::ComputeType(dtype::DType::U8), dtype::DType::I32);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::U16), dtype::DType::I32);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::U32), dtype::DType::I32);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::U64), dtype::DType::I64);
    // floating_point
    EXPECT_EQ(dtype::ComputeType(dtype::DType::F8E4M3), dtype::DType::F16);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::F8E5M2), dtype::DType::F16);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::F16), dtype::DType::F32);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::F32), dtype::DType::F32);
    EXPECT_EQ(dtype::ComputeType(dtype::DType::F64), dtype::DType::F64);
}

TEST(DTypeComputeType, AllTypesHaveComputeType) {
    // パラメーターテスト: すべてのdtypeで計算用型が正しいか
    for (const auto dtype : dtype::kAllDTypes) {
        const auto compute_type = dtype::ComputeType(dtype);
        const auto compute_index = dtype::ToIndex(compute_type);
        EXPECT_LT(compute_index, dtype::kDTypeCount);
    }
}

TEST(DTypeComputeType, Idempotency) {
    // 動作の順序: 計算用型の一貫性（べき等性）
    for (const auto dtype : dtype::kAllDTypes) {
        const auto compute_type = dtype::ComputeType(dtype);
        EXPECT_EQ(dtype::ComputeType(compute_type), compute_type);
    }
}
