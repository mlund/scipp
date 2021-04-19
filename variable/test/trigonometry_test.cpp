// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/common/constants.h"
#include "scipp/variable/creation.h"
#include "scipp/variable/to_unit.h"
#include "scipp/variable/trigonometry.h"

using namespace scipp;
using namespace scipp::variable;
using namespace scipp::units;

class VariableTrigonometryTest : public ::testing::Test {
protected:
  [[nodiscard]] static Variable input_in_rad() {
    return makeVariable<double>(Dims{Dim::X}, Shape{7},
                                Values{0.0, pi<double> / 2.0, pi<double>,
                                       -pi<double> * 3.0 / 2.0,
                                       2.0 * pi<double>, -0.123, 1.654},
                                Unit{units::rad});
  }

  [[nodiscard]] static Variable input_in_deg() {
    return to_unit(input_in_rad(), units::deg);
  }

  [[nodiscard]] static Variable expected_for_op(double (*op)(const double)) {
    std::vector<double> res;
    const auto var = input_in_rad();
    auto values_in_rad = var.values<double>();
    std::transform(values_in_rad.begin(), values_in_rad.end(),
                   back_inserter(res),
                   [&op](double x_in_rad) { return op(x_in_rad); });
    return makeVariable<double>(Dims{Dim::X}, Shape{res.size()}, Values(res));
  }
};

TEST_F(VariableTrigonometryTest, sin_rad) {
  const auto var = input_in_rad();
  EXPECT_EQ(sin(var), expected_for_op(std::sin));
  EXPECT_EQ(var, input_in_rad());
}

TEST_F(VariableTrigonometryTest, sin_deg) {
  const auto var = input_in_deg();
  EXPECT_EQ(sin(var), expected_for_op(std::sin));
  EXPECT_EQ(var, input_in_deg());
}

TEST_F(VariableTrigonometryTest, sin_move_rad) {
  auto var = input_in_rad();
  const auto ptr = var.values<double>().data();
  auto out = sin(std::move(var));
  EXPECT_EQ(out, expected_for_op(std::sin));
  EXPECT_EQ(out.values<double>().data(), ptr);
}

TEST_F(VariableTrigonometryTest, sin_move_deg) {
  auto var = input_in_deg();
  const auto ptr = var.values<double>().data();
  auto out = sin(std::move(var));
  EXPECT_EQ(out, expected_for_op(std::sin));
  EXPECT_EQ(out.values<double>().data(), ptr);
}

TEST_F(VariableTrigonometryTest, sin_out_arg_rad) {
  const auto in = input_in_rad();
  auto out = special_like(in, FillValue::ZeroNotBool);
  auto &view = sin(in, out);

  EXPECT_EQ(out, expected_for_op(std::sin));
  EXPECT_EQ(&view, &out);
  EXPECT_EQ(in, input_in_rad());
}

TEST_F(VariableTrigonometryTest, sin_out_arg_deg) {
  const auto in = input_in_deg();
  auto out = special_like(in, FillValue::ZeroNotBool);
  auto &view = sin(in, out);

  EXPECT_EQ(out, expected_for_op(std::sin));
  EXPECT_EQ(&view, &out);
  EXPECT_EQ(in, input_in_deg());
}

TEST_F(VariableTrigonometryTest, cos_rad) {
  const auto var = input_in_rad();
  EXPECT_EQ(cos(var), expected_for_op(std::cos));
  EXPECT_EQ(var, input_in_rad());
}

TEST_F(VariableTrigonometryTest, cos_deg) {
  const auto var = input_in_deg();
  EXPECT_EQ(cos(var), expected_for_op(std::cos));
  EXPECT_EQ(var, input_in_deg());
}

TEST_F(VariableTrigonometryTest, cos_move_rad) {
  auto var = input_in_rad();
  const auto ptr = var.values<double>().data();
  auto out = cos(std::move(var));
  EXPECT_EQ(out, expected_for_op(std::cos));
  EXPECT_EQ(out.values<double>().data(), ptr);
}

TEST_F(VariableTrigonometryTest, cos_move_deg) {
  auto var = input_in_deg();
  const auto ptr = var.values<double>().data();
  auto out = cos(std::move(var));
  EXPECT_EQ(out, expected_for_op(std::cos));
  EXPECT_EQ(out.values<double>().data(), ptr);
}

TEST_F(VariableTrigonometryTest, cos_out_arg_rad) {
  const auto in = input_in_rad();
  auto out = special_like(in, FillValue::ZeroNotBool);
  auto &view = cos(in, out);

  EXPECT_EQ(out, expected_for_op(std::cos));
  EXPECT_EQ(&view, &out);
  EXPECT_EQ(in, input_in_rad());
}

TEST_F(VariableTrigonometryTest, cos_out_arg_deg) {
  const auto in = input_in_deg();
  auto out = special_like(in, FillValue::ZeroNotBool);
  auto &view = cos(in, out);

  EXPECT_EQ(out, expected_for_op(std::cos));
  EXPECT_EQ(&view, &out);
  EXPECT_EQ(in, input_in_deg());
}

TEST_F(VariableTrigonometryTest, tan_rad) {
  const auto var = input_in_rad();
  EXPECT_EQ(tan(var), expected_for_op(std::tan));
  EXPECT_EQ(var, input_in_rad());
}

TEST_F(VariableTrigonometryTest, tan_deg) {
  const auto var = input_in_deg();
  EXPECT_EQ(tan(var), expected_for_op(std::tan));
  EXPECT_EQ(var, input_in_deg());
}

TEST_F(VariableTrigonometryTest, tan_move_rad) {
  auto var = input_in_rad();
  const auto ptr = var.values<double>().data();
  auto out = tan(std::move(var));
  EXPECT_EQ(out, expected_for_op(std::tan));
  EXPECT_EQ(out.values<double>().data(), ptr);
}

TEST_F(VariableTrigonometryTest, tan_move_deg) {
  auto var = input_in_deg();
  const auto ptr = var.values<double>().data();
  auto out = tan(std::move(var));
  EXPECT_EQ(out, expected_for_op(std::tan));
  EXPECT_EQ(out.values<double>().data(), ptr);
}

TEST_F(VariableTrigonometryTest, tan_out_arg_rad) {
  const auto in = input_in_rad();
  auto out = special_like(in, FillValue::ZeroNotBool);
  auto &view = tan(in, out);

  EXPECT_EQ(out, expected_for_op(std::tan));
  EXPECT_EQ(&view, &out);
  EXPECT_EQ(in, input_in_rad());
}

TEST_F(VariableTrigonometryTest, tan_out_arg_deg) {
  const auto in = input_in_deg();
  auto out = special_like(in, FillValue::ZeroNotBool);
  auto &view = tan(in, out);

  EXPECT_EQ(out, expected_for_op(std::tan));
  EXPECT_EQ(&view, &out);
  EXPECT_EQ(in, input_in_deg());
}

TEST_F(VariableTrigonometryTest, asin) {
  const auto var = makeVariable<double>(Values{1.0});
  EXPECT_EQ(asin(var),
            makeVariable<double>(Values{std::asin(1.0)}, Unit{units::rad}));
}

TEST_F(VariableTrigonometryTest, asin_move) {
  auto var = makeVariable<double>(Values{1.0});
  const auto ptr = var.values<double>().data();
  auto out = asin(std::move(var));
  EXPECT_EQ(out,
            makeVariable<double>(Values{std::asin(1.0)}, Unit{units::rad}));
  EXPECT_EQ(out.values<double>().data(), ptr);
}

TEST_F(VariableTrigonometryTest, asin_out_arg) {
  auto x = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, 0.0});
  auto out = makeVariable<double>(Values{0.0});
  auto &view = asin(x.slice({Dim::X, 0}), out);

  EXPECT_EQ(out,
            makeVariable<double>(Values{std::asin(1.0)}, Unit{units::rad}));
  EXPECT_EQ(&view, &out);
}

TEST_F(VariableTrigonometryTest, acos) {
  const auto var = makeVariable<double>(Values{1.0});
  EXPECT_EQ(acos(var),
            makeVariable<double>(Values{std::acos(1.0)}, Unit{units::rad}));
}

TEST_F(VariableTrigonometryTest, acos_move) {
  auto var = makeVariable<double>(Values{1.0});
  const auto ptr = var.values<double>().data();
  auto out = acos(std::move(var));
  EXPECT_EQ(out,
            makeVariable<double>(Values{std::acos(1.0)}, Unit{units::rad}));
  EXPECT_EQ(out.values<double>().data(), ptr);
}

TEST_F(VariableTrigonometryTest, acos_out_arg) {
  auto x = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, 0.0});
  auto out = makeVariable<double>(Values{0.0});
  auto &view = acos(x.slice({Dim::X, 0}), out);

  EXPECT_EQ(out,
            makeVariable<double>(Values{std::acos(1.0)}, Unit{units::rad}));
  EXPECT_EQ(&view, &out);
}

TEST_F(VariableTrigonometryTest, atan) {
  const auto var = makeVariable<double>(Values{1.0});
  EXPECT_EQ(atan(var),
            makeVariable<double>(Values{std::atan(1.0)}, Unit{units::rad}));
}

TEST_F(VariableTrigonometryTest, atan_move) {
  auto var = makeVariable<double>(Values{1.0});
  const auto ptr = var.values<double>().data();
  auto out = atan(std::move(var));
  EXPECT_EQ(out,
            makeVariable<double>(Values{std::atan(1.0)}, Unit{units::rad}));
  EXPECT_EQ(out.values<double>().data(), ptr);
}

TEST_F(VariableTrigonometryTest, atan_out_arg) {
  auto x = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, 0.0});
  auto out = makeVariable<double>(Values{0.0});
  auto &view = atan(x.slice({Dim::X, 0}), out);

  EXPECT_EQ(out,
            makeVariable<double>(Values{std::atan(1.0)}, Unit{units::rad}));
  EXPECT_EQ(&view, &out);
}

TEST_F(VariableTrigonometryTest, atan2) {
  auto x = makeVariable<double>(units::m, Values{1.0});
  auto y = x;
  auto expected =
      makeVariable<double>(units::rad, Values{scipp::pi<double> / 4});
  EXPECT_EQ(atan2(y, x), expected);
}

TEST_F(VariableTrigonometryTest, atan2_out_arg) {
  auto x = makeVariable<double>(units::m, Values{1.0});
  auto y = x;
  auto expected =
      makeVariable<double>(units::rad, Values{scipp::pi<double> / 4});
  auto out = atan2(y, x, y);
  EXPECT_EQ(out, expected);
  EXPECT_EQ(y, expected);
}
