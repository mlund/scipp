// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/core/dimensions.h"
#include "scipp/core/except.h"
#include "scipp/core/slice.h"
#include "scipp/dataset/dataset.h"
#include "scipp/dataset/slice.h"
#include "scipp/variable/arithmetic.h"
#include "test_macros.h"

using namespace scipp;

namespace {
template <int N>
constexpr auto make_array_common =
    [](const auto... values) {
      const auto size = sizeof...(values);
      Variable coord = makeVariable<double>(units::m, Dims{Dim::X}, Shape{size},
                                            Values{values...});
      Variable data = makeVariable<int64_t>(Dims{Dim::X}, Shape{size - N});
      return DataArray{data, {{Dim::X, coord}}};
    };
}
constexpr auto make_points = make_array_common<0>;
constexpr auto make_histogram = make_array_common<1>;

TEST(SliceByValueTest, test_dimension_not_found) {
  auto var =
      makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1.0, 2.0, 3.0, 4.0});
  DataArray da{var, {{Dim::X, var}}};
  EXPECT_THROW(auto s = slice(da, Dim::Y), except::NotFoundError);
}

TEST(SliceByValueTest, test_no_multi_dimensional_coords) {
  auto var = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});
  DataArray da{var, {{Dim::X, var}}};
  EXPECT_THROW(auto s = slice(da, Dim::X), except::DimensionError);
}

TEST(SliceByValueTest, test_unsorted_coord_throws) {
  auto unsorted =
      makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1.0, 2.0, 3.0, 1.5});
  DataArray da{unsorted, {{Dim::X, unsorted}}};
  EXPECT_THROW(auto s = slice(da, Dim::X), std::runtime_error);
}

TEST(SliceByValueTest, test_begin_end_not_0D_throws) {
  auto da = make_points(0, 1, 2, 3);
  auto one_d = makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{1.0});
  EXPECT_THROW(auto s = slice(da, Dim::X, one_d, VariableConstView{}),
               except::MismatchError<Dimensions>);
  EXPECT_THROW(auto s = slice(da, Dim::X, VariableConstView{}, one_d),
               except::MismatchError<Dimensions>);
}

TEST(SliceByValueTest, test_slicing_defaults_ascending) {
  auto da = make_points(3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  EXPECT_EQ(da, slice(da, Dim::X, VariableConstView{}, 13.0 * units::m));
  EXPECT_EQ(da, slice(da, Dim::X));
}

TEST(SliceByValueTest, test_slicing_defaults_descending) {
  auto da = make_points(12, 11, 10, 9, 8, 7, 6, 5, 4, 3);
  EXPECT_EQ(da, slice(da, Dim::X, VariableConstView{}, 2.0 * units::m));
  EXPECT_EQ(da, slice(da, Dim::X));
}

TEST(SliceByValueTest, test_slice_range_on_point_coords_1D_ascending) {
  auto da = make_points(3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  // No effect slicing
  auto out = slice(da, Dim::X, 3.0 * units::m, 13.0 * units::m);
  EXPECT_EQ(da, out);
  // Test start on left boundary (closed on left), so includes boundary
  out = slice(da, Dim::X, 3.0 * units::m, 4.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 0, 1}));
  // Test start out of bounds on left truncated
  out = slice(da, Dim::X, 2.0 * units::m, 4.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 0, 1}));
  // Test inner values
  out = slice(da, Dim::X, 3.5 * units::m, 5.5 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 1, 3}));
  // Test end on right boundary (open on right), so does not include boundary
  out = slice(da, Dim::X, 11.0 * units::m, 12.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 8, 9}));
  // Test end out of bounds on right truncated
  out = slice(da, Dim::X, 11.0 * units::m, 13.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 8, 10}));
}

TEST(SliceByValueTest, test_slice_range_on_point_coords_1D_descending) {
  auto da = make_points(12, 11, 10, 9, 8, 7, 6, 5, 4, 3);
  // No effect slicing
  auto out = slice(da, Dim::X, 12.0 * units::m, 2.0 * units::m);
  EXPECT_EQ(da, out);
  // Test start on left boundary (closed on left), so includes boundary
  out = slice(da, Dim::X, 12.0 * units::m, 11.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 0, 1}));
  // Test start out of bounds on left truncated
  out = slice(da, Dim::X, 13.0 * units::m, 11.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 0, 1}));
  // Test inner values
  out = slice(da, Dim::X, 11.5 * units::m, 9.5 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 1, 3}));
  // Test end on right boundary (open on right), so does not include boundary
  out = slice(da, Dim::X, 4.0 * units::m, 3.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 8, 9}));
  // Test end out of bounds on right truncated
  out = slice(da, Dim::X, 4.0 * units::m, 1.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 8, 10}));
}

TEST(SliceByValueTest, test_slice_range_on_edge_coords_1D_ascending) {
  auto da = make_histogram(3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  // No effect slicing
  auto out = slice(da, Dim::X, 3.0 * units::m, 13.0 * units::m);
  EXPECT_EQ(out, da);
  // Test start on left boundary (closed on left), so includes boundary
  out = slice(da, Dim::X, 3.0 * units::m, 4.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 0, 1}));
  // Test slicing with range boundary inside edge, same result as above expected
  out = slice(da, Dim::X, 3.1 * units::m, 4.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 0, 1}));
  // Test slicing with range lower boundary on upper edge of bin (open on right
  // test)
  out = slice(da, Dim::X, 4.0 * units::m, 6.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 1, 3}));
  // Test end on right boundary (open on right), so does not include boundary
  out = slice(da, Dim::X, 11.0 * units::m, 12.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 8, 9}));
}

TEST(SliceByValueTest, test_slice_range_on_edge_coords_1D_descending) {
  auto da = make_histogram(12, 11, 10, 9, 8, 7, 6, 5, 4, 3);
  // No effect slicing
  auto out = slice(da, Dim::X, 12.0 * units::m, 2.0 * units::m);
  EXPECT_EQ(out, da);
  // Test start on left boundary (closed on left), so includes boundary
  out = slice(da, Dim::X, 12.0 * units::m, 11.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 0, 1}));
  // Test slicing with range boundary inside edge, same result as above expected
  out = slice(da, Dim::X, 11.9 * units::m, 11.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 0, 1}));
  // Test slicing with range lower boundary on upper edge of bin (open on right
  // test)
  out = slice(da, Dim::X, 11.0 * units::m, 9.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 1, 3}));
  // Test end on right boundary (open on right), so does not include boundary
  out = slice(da, Dim::X, 4.0 * units::m, 3.0 * units::m);
  EXPECT_EQ(out, da.slice({Dim::X, 8, 9}));
}

TEST(SliceByValueTest, test_point_on_point_coords_1D_ascending) {
  auto da = make_points(3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  // No effect slicing
  // Test start on left boundary (closed on left), so includes boundary
  auto begin = 3.0 * units::m;
  auto out = slice(da, Dim::X, begin, begin);
  EXPECT_EQ(out, da.slice({Dim::X, 0}));
  // Test point slice between points throws
  begin = 3.5 * units::m;
  EXPECT_THROW(auto s = slice(da, Dim::X, begin, begin), except::NotFoundError);
  // Test start on right boundary
  begin = 12.0 * units::m;
  out = slice(da, Dim::X, begin, begin);
  EXPECT_EQ(out, da.slice({Dim::X, 9}));
  // Test start outside right boundary throws
  begin = 12.1 * units::m;
  EXPECT_THROW(auto s = slice(da, Dim::X, begin, begin), except::NotFoundError);
}

TEST(SliceByValueTest, test_point_on_point_coords_1D_descending) {
  auto da = make_points(12, 11, 10, 9, 8, 7, 6, 5, 4, 3);
  // No effect slicing
  // Test start on left boundary (closed on left), so includes boundary
  auto begin = 12.0 * units::m;
  auto out = slice(da, Dim::X, begin, begin);
  EXPECT_EQ(out, da.slice({Dim::X, 0}));
  // Test point slice between points throws
  begin = 3.5 * units::m;
  EXPECT_THROW(auto s = slice(da, Dim::X, begin, begin), except::NotFoundError);
  // Test start on right boundary
  begin = 3.0 * units::m;
  out = slice(da, Dim::X, begin, begin);
  EXPECT_EQ(out, da.slice({Dim::X, 9}));
  // Test start outside right boundary throws
  begin = 2.99 * units::m;
  EXPECT_THROW(auto s = slice(da, Dim::X, begin, begin), except::NotFoundError);
}

TEST(SliceByValueTest, test_slice_point_on_edge_coords_1D) {
  auto da = make_histogram(3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  // test no-effect slicing
  // Test start on left boundary (closed on left), so includes boundary
  auto begin = 3.0 * units::m;
  auto out = slice(da, Dim::X, begin, begin);
  EXPECT_EQ(out, da.slice({Dim::X, 0}));
  // Same as above, takes lower bounds of bin so same bin
  begin = 3.5 * units::m;
  out = slice(da, Dim::X, begin, begin);
  EXPECT_EQ(out, da.slice({Dim::X, 0}));
  // Next bin
  begin = 4.0 * units::m;
  out = slice(da, Dim::X, begin, begin);
  EXPECT_EQ(out, da.slice({Dim::X, 1}));
  // Last bin
  begin = 11.9 * units::m;
  out = slice(da, Dim::X, begin, begin);
  EXPECT_EQ(out, da.slice({Dim::X, 8}));
  // (closed on right) so out of bounds
  begin = 12.0 * units::m;
  EXPECT_THROW(auto s = slice(da, Dim::X, begin, begin), except::NotFoundError);
  // out of bounds for left for completeness
  begin = 2.99 * units::m;
  EXPECT_THROW(auto s = slice(da, Dim::X, begin, begin), except::NotFoundError);
}
