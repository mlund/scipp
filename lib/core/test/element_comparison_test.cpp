// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/core/element/comparison.h"
#include "scipp/units/unit.h"

#include "fix_typed_test_suite_warnings.h"
#include "test_macros.h"

using namespace scipp;
using namespace scipp::core::element;

template <typename T> class ElementLessTest : public ::testing::Test {};
template <typename T> class ElementGreaterTest : public ::testing::Test {};
template <typename T> class ElementLessEqualTest : public ::testing::Test {};
template <typename T> class ElementGreaterEqualTest : public ::testing::Test {};
template <typename T> class ElementEqualTest : public ::testing::Test {};
template <typename T> class ElementNotEqualTest : public ::testing::Test {};
using ElementLessTestTypes = ::testing::Types<double, float, int64_t, int32_t>;
TYPED_TEST_SUITE(ElementLessTest, ElementLessTestTypes);
TYPED_TEST_SUITE(ElementGreaterTest, ElementLessTestTypes);
TYPED_TEST_SUITE(ElementLessEqualTest, ElementLessTestTypes);
TYPED_TEST_SUITE(ElementGreaterEqualTest, ElementLessTestTypes);
TYPED_TEST_SUITE(ElementEqualTest, ElementLessTestTypes);
TYPED_TEST_SUITE(ElementNotEqualTest, ElementLessTestTypes);

TEST(ElementComparisonTest, unit) {
  const units::Unit m(units::m);
  EXPECT_EQ(comparison(m, m), units::none);
  const units::Unit rad(units::rad);
  EXPECT_THROW(comparison(rad, m), except::UnitError);
}

TYPED_TEST(ElementLessTest, value) {
  using T = TypeParam;
  T y = 1;
  T x = 2;
  EXPECT_EQ(less(y, x), true);
  x = -1;
  EXPECT_EQ(less(y, x), false);
  x = 1;
  EXPECT_EQ(less(y, x), false);
}

TYPED_TEST(ElementGreaterTest, value) {
  using T = TypeParam;
  T y = 1;
  T x = 2;
  EXPECT_EQ(greater(y, x), false);
  x = -1;
  EXPECT_EQ(greater(y, x), true);
  x = 1;
  EXPECT_EQ(greater(y, x), false);
}

TYPED_TEST(ElementLessEqualTest, value) {
  using T = TypeParam;
  T y = 1;
  T x = 2;
  EXPECT_EQ(less_equal(y, x), true);
  x = 1;
  EXPECT_EQ(less_equal(y, x), true);
  x = -1;
  EXPECT_EQ(less_equal(y, x), false);
}

TYPED_TEST(ElementGreaterEqualTest, value) {
  using T = TypeParam;
  T y = 1;
  T x = 2;
  EXPECT_EQ(greater_equal(y, x), false);
  x = 1;
  EXPECT_EQ(greater_equal(y, x), true);
  x = -1;
  EXPECT_EQ(greater_equal(y, x), true);
}

TYPED_TEST(ElementEqualTest, value) {
  using T = TypeParam;
  T y = 1;
  T x = 2;
  EXPECT_EQ(equal(y, x), false);
  x = 1;
  EXPECT_EQ(equal(y, x), true);
  x = -1;
  EXPECT_EQ(equal(y, x), false);
}

TYPED_TEST(ElementNotEqualTest, value) {
  using T = TypeParam;
  T y = 1;
  T x = 2;
  EXPECT_EQ(not_equal(y, x), true);
  x = 1;
  EXPECT_EQ(not_equal(y, x), false);
  x = -1;
  EXPECT_EQ(not_equal(y, x), true);
}

template <typename T> class ElementNanMinTest : public ::testing::Test {};
template <typename T> class ElementNanMaxTest : public ::testing::Test {};
using ElementNanMinTestTypes = ::testing::Types<double, float>;
TYPED_TEST_SUITE(ElementNanMinTest, ElementNanMinTestTypes);
TYPED_TEST_SUITE(ElementNanMaxTest, ElementNanMinTestTypes);

TYPED_TEST(ElementNanMinTest, value) {
  using T = TypeParam;
  T y = 1;
  T x = 2;
  nanmin_equals(y, x);
  EXPECT_EQ(y, 1);
}

TYPED_TEST(ElementNanMinTest, value_nan) {
  using T = TypeParam;
  T y = NAN;
  T x = 2;
  nanmin_equals(y, x);
  EXPECT_EQ(y, 2);
}

TYPED_TEST(ElementNanMaxTest, value) {
  using T = TypeParam;
  T y = 1;
  T x = 2;
  nanmax_equals(y, x);
  EXPECT_EQ(y, 2);
}

TYPED_TEST(ElementNanMaxTest, value_nan) {
  using T = TypeParam;
  T y = 1;
  T x = NAN;
  nanmax_equals(y, x);
  EXPECT_EQ(y, 1);
}

template <typename T> class IsCloseTest : public ::testing::Test {};
using IsCloseTestTypes = ::testing::Types<double, ValueAndVariance<double>>;
TYPED_TEST_SUITE(IsCloseTest, IsCloseTestTypes);

TYPED_TEST(IsCloseTest, value) {
  TypeParam a = 1.0;
  TypeParam b = 2.1;
  EXPECT_TRUE(isclose(a, b, 1.2));
  EXPECT_TRUE(isclose(a, b, 1.1));
  EXPECT_FALSE(isclose(a, b, 1.0));
}

TYPED_TEST(IsCloseTest, value_not_equal_nans) {
  EXPECT_FALSE(isclose(TypeParam(NAN), TypeParam(NAN), 1.e9));
  EXPECT_FALSE(isclose(TypeParam(NAN), TypeParam(1.0), 1.e9));
  EXPECT_FALSE(isclose(TypeParam(1.0), TypeParam(NAN), 1.e9));
  EXPECT_FALSE(isclose(TypeParam(INFINITY), TypeParam(INFINITY), 1.e9));
  EXPECT_FALSE(isclose(TypeParam(1.0), TypeParam(INFINITY), 1.e9));
  EXPECT_FALSE(isclose(TypeParam(INFINITY), TypeParam(1.0), 1.e9));
  EXPECT_FALSE(isclose(-TypeParam(INFINITY), -TypeParam(INFINITY), 1.e9));
  EXPECT_FALSE(isclose(-TypeParam(1.0), -TypeParam(INFINITY), 1.e9));
  EXPECT_FALSE(isclose(-TypeParam(INFINITY), -TypeParam(1.0), 1.e9));
}

TYPED_TEST(IsCloseTest, value_equal_nans) {
  EXPECT_TRUE(isclose_equal_nan(TypeParam(NAN), TypeParam(NAN), 1.e9));
  EXPECT_FALSE(isclose_equal_nan(TypeParam(NAN), TypeParam(1.0), 1.e9));
  EXPECT_FALSE(isclose_equal_nan(TypeParam(1.0), TypeParam(NAN), 1.e9));
}
TYPED_TEST(IsCloseTest, value_equal_pos_infs) {
  EXPECT_TRUE(
      isclose_equal_nan(TypeParam(INFINITY), TypeParam(INFINITY), 1.e9));
  EXPECT_FALSE(isclose_equal_nan(TypeParam(1.0), TypeParam(INFINITY), 1.e9));
  EXPECT_FALSE(isclose_equal_nan(TypeParam(INFINITY), TypeParam(1.0), 1.e9));
}
TYPED_TEST(IsCloseTest, value_equal_neg_infs) {
  EXPECT_TRUE(
      isclose_equal_nan(-TypeParam(INFINITY), -TypeParam(INFINITY), 1.e9));
  EXPECT_FALSE(isclose_equal_nan(-TypeParam(1.0), -TypeParam(INFINITY), 1.e9));
  EXPECT_FALSE(isclose_equal_nan(-TypeParam(INFINITY), -TypeParam(1.0), 1.e9));
}

TYPED_TEST(IsCloseTest, value_equal_infs_signbit) {
  EXPECT_FALSE(
      isclose_equal_nan(-TypeParam(INFINITY), TypeParam(INFINITY), 1.e9));
  EXPECT_FALSE(
      isclose_equal_nan(TypeParam(INFINITY), -TypeParam(INFINITY), 1.e9));
}

template <class Op> void do_isclose_units_test(Op op) {
  EXPECT_EQ(units::none, op(units::m, units::m, units::m));
  EXPECT_THROW_DISCARD(op(units::m, units::m, units::s), except::UnitError);
  EXPECT_THROW_DISCARD(op(units::m, units::s, units::m), except::UnitError);
  EXPECT_THROW_DISCARD(op(units::s, units::m, units::m), except::UnitError);
}

TEST(IsCloseTest, units) {
  do_isclose_units_test(isclose);
  do_isclose_units_test(isclose_equal_nan);
}

constexpr auto check_inplace = [](auto op, auto a, auto b, auto expected) {
  op(a, b);
  EXPECT_EQ(a, expected);
};

TEST(ComparisonTest, min_max_support_time_point) {
  static_cast<void>(std::get<core::time_point>(decltype(max_equals)::types{}));
  static_cast<void>(std::get<core::time_point>(decltype(min_equals)::types{}));
  static_cast<void>(
      std::get<core::time_point>(decltype(nanmax_equals)::types{}));
  static_cast<void>(
      std::get<core::time_point>(decltype(nanmin_equals)::types{}));
}

TEST(ComparisonTest, max_equals) {
  check_inplace(max_equals, 1, 2, 2);
  check_inplace(max_equals, 2, 1, 2);
  check_inplace(max_equals, 1.2, 1.3, 1.3);
  check_inplace(max_equals, 1.3, 1.2, 1.3);
  check_inplace(max_equals, core::time_point(23), core::time_point(13),
                core::time_point(23));
}

TEST(ComparisonTest, min_equals) {
  check_inplace(min_equals, 1, 2, 1);
  check_inplace(min_equals, 2, 1, 1);
  check_inplace(min_equals, 1.2, 1.3, 1.2);
  check_inplace(min_equals, 1.3, 1.2, 1.2);
  check_inplace(min_equals, core::time_point(23), core::time_point(13),
                core::time_point(13));
}
