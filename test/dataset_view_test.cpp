/// @file
/// SPDX-License-Identifier: GPL-3.0-or-later
/// @author Simon Heybrock
/// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory, NScD Oak Ridge
/// National Laboratory, and European Spallation Source ERIC.
#include <gtest/gtest.h>

#include <boost/mpl/at.hpp>
#include <boost/mpl/sort.hpp>
#include <boost/mpl/vector_c.hpp>

#include "test_macros.h"

#include "dataset_view.h"

TEST(DatasetView, construct) {
  Dataset d;
  d.insert<Data::Value>("name1", Dimensions{}, {1.1});
  d.insert<Data::Int>("name2", Dimensions{}, {2});
  // Empty view forbidden by static_assert:
  // DatasetView<> view(d);
  ASSERT_NO_THROW(static_cast<void>(DatasetView<Data::Value>(d)));
  ASSERT_NO_THROW(static_cast<void>(DatasetView<Data::Int>(d)));
  ASSERT_NO_THROW(static_cast<void>(DatasetView<Data::Int, Data::Value>(d)));
  ASSERT_THROW(static_cast<void>(DatasetView<Data::Int, Data::Variance>(d)),
               std::runtime_error);
}

TEST(DatasetView, construct_with_const_Dataset) {
  Dataset d;
  d.insert<Data::Value>("name1", {Dim::X, 1}, {1.1});
  d.insert<Data::Int>("name2", Dimensions{}, {2});
  const auto const_d(d);
  EXPECT_NO_THROW(DatasetView<const Data::Value> view(const_d));
  EXPECT_NO_THROW(
      DatasetView<DatasetView<const Data::Value>> nested(const_d, {Dim::X}));
  EXPECT_NO_THROW(static_cast<void>(
      DatasetView<DatasetView<const Data::Value>, const Data::Int>(const_d,
                                                                   {Dim::X})));
}

TEST(DatasetView, iterator) {
  Dataset d;
  d.insert<Data::Value>("name1", Dimensions{Dim::X, 2}, {1.1, 1.2});
  d.insert<Data::Int>("name2", Dimensions{Dim::X, 2}, {2, 3});
  DatasetView<Data::Value> view(d);
  ASSERT_NO_THROW(view.begin());
  ASSERT_NO_THROW(view.end());
  auto it = view.begin();
  // Note: Cannot assigned dereferenced iterator by value since Dataset::Item
  // should not live outside and iterator.
  // auto item = *it;
  ASSERT_EQ(it->get<Data::Value>(), 1.1);
  it->get<Data::Value>() = 2.2;
  ASSERT_EQ(it->value(), 2.2);
  ASSERT_EQ(it, it);
  ASSERT_EQ(it, view.begin());
  ASSERT_NE(it, view.end());
  ASSERT_NO_THROW(it++);
  ASSERT_NE(it, view.end());
  ASSERT_EQ(it->value(), 1.2);
  ASSERT_NO_THROW(it++);
  ASSERT_EQ(it, view.end());
}

TEST(DatasetView, copy_on_write) {
  Dataset d;
  d.insert<Coord::X>({Dim::X, 2}, 2);
  d.insert<Coord::Y>({Dim::X, 2}, 2);
  const auto copy(d);

  DatasetView<const Coord::X> const_view(d);
  EXPECT_EQ(&const_view.begin()->get<Coord::X>(),
            &copy.get<const Coord::X>()[0]);
  // Again, just to confirm that the call to `copy.get` is not the reason for
  // breaking sharing:
  EXPECT_EQ(&const_view.begin()->get<Coord::X>(),
            &copy.get<const Coord::X>()[0]);

  DatasetView<Coord::X, const Coord::Y> view(d);
  EXPECT_NE(&view.begin()->get<Coord::X>(), &copy.get<const Coord::X>()[0]);
  // Breaks sharing only for the non-const variables:
  EXPECT_EQ(&view.begin()->get<Coord::Y>(), &copy.get<const Coord::Y>()[0]);
}

TEST(DatasetView, single_column) {
  Dataset d;
  d.insert<Data::Value>("name1", Dimensions(Dim::Tof, 10), 10);
  d.insert<Data::Int>("name2", Dimensions(Dim::Tof, 10), 10);
  auto var = d.get<Data::Value>();
  var[0] = 0.2;
  var[3] = 3.2;

  DatasetView<Data::Value> view(d);
  auto it = view.begin();
  ASSERT_EQ(it->get<Data::Value>(), 0.2);
  it++;
  ASSERT_EQ(it->get<Data::Value>(), 0.0);
  it++;
  ASSERT_EQ(it->get<Data::Value>(), 0.0);
  it++;
  ASSERT_EQ(it->get<Data::Value>(), 3.2);
  it += 7;
  ASSERT_EQ(it, view.end());
}

TEST(DatasetView, multi_column) {
  Dataset d;
  d.insert<Data::Value>("name1", Dimensions(Dim::Tof, 2), 2);
  d.insert<Data::Int>("name2", Dimensions(Dim::Tof, 2), 2);
  auto var = d.get<Data::Value>();
  var[0] = 0.2;
  var[1] = 3.2;

  DatasetView<Data::Value, Data::Int> view(d);
  auto it = view.begin();
  ASSERT_EQ(it->get<Data::Value>(), 0.2);
  ASSERT_EQ(it->get<Data::Int>(), 0);
  it++;
  ASSERT_EQ(it->get<Data::Value>(), 3.2);
  ASSERT_EQ(it->get<Data::Int>(), 0);
}

TEST(DatasetView, multi_column_mixed_dimension) {
  Dataset d;
  d.insert<Data::Value>("name1", Dimensions(Dim::Tof, 2), 2);
  d.insert<Data::Int>("name2", Dimensions{}, 1);
  auto var = d.get<Data::Value>();
  var[0] = 0.2;
  var[1] = 3.2;

  ASSERT_ANY_THROW(static_cast<void>(DatasetView<Data::Value, Data::Int>(d)));
  ASSERT_NO_THROW(
      static_cast<void>(DatasetView<Data::Value, const Data::Int>(d)));
  auto view = (DatasetView<Data::Value, const Data::Int>(d));
  auto it = view.begin();
  ASSERT_EQ(it->get<Data::Value>(), 0.2);
  ASSERT_EQ(it->get<Data::Int>(), 0);
  it++;
  ASSERT_EQ(it->get<Data::Value>(), 3.2);
  ASSERT_EQ(it->get<Data::Int>(), 0);
}

TEST(DatasetView, multi_column_transposed) {
  Dataset d;
  Dimensions dimsXY;
  dimsXY.add(Dim::X, 2);
  dimsXY.add(Dim::Y, 3);
  Dimensions dimsYX;
  dimsYX.add(Dim::Y, 3);
  dimsYX.add(Dim::X, 2);

  d.insert<Data::Value>("name1", dimsXY, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  d.insert<Data::Int>("name2", dimsYX, {1, 3, 5, 2, 4, 6});
  // TODO Current dimension check is too strict and fails unless data with
  // transposed dimensions is accessed as const.
  DatasetView<Data::Value, const Data::Int> view(d);
  auto it = view.begin();
  ASSERT_NE(++it, view.end());
  ASSERT_EQ(it->get<Data::Value>(), 2.0);
  ASSERT_EQ(it->get<Data::Int>(), 2);
  for (const auto &item : view)
    ASSERT_EQ(item.get<Data::Value>(), item.get<Data::Int>());
}

TEST(DatasetView, multi_column_unrelated_dimension) {
  Dataset d;
  d.insert<Data::Value>("name1", Dimensions(Dim::X, 2), 2);
  d.insert<Data::Int>("name2", Dimensions(Dim::Y, 3), 3);
  DatasetView<Data::Value> view(d);
  auto it = view.begin();
  ASSERT_TRUE(it < view.end());
  it += 2;
  // We iterate only Data::Value, so there should be no iteration in
  // Dim::Y.
  ASSERT_EQ(it, view.end());
}

TEST(DatasetView, multi_column_orthogonal_fail) {
  Dataset d;
  d.insert<Data::Value>("name1", Dimensions(Dim::X, 2), 2);
  d.insert<Data::Int>("name2", Dimensions(Dim::Y, 3), 3);
  EXPECT_THROW_MSG((DatasetView<Data::Value, Data::Int>(d)), std::runtime_error,
                   "Variables requested for iteration do not span a joint "
                   "space. In case one of the variables represents bin edges "
                   "direct joint iteration is not possible. Use the Bin<> "
                   "wrapper to iterate over bins defined by edges instead.");
}

TEST(DatasetView, nested_DatasetView) {
  Dataset d;
  d.insert<Data::Value>("name1", {{Dim::Y, 3}, {Dim::X, 2}},
                        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  d.insert<Data::Int>("name2", {Dim::X, 2}, {10, 20});
  DatasetView<DatasetView<const Data::Value>, const Data::Int> view(d,
                                                                    {Dim::Y});
  ASSERT_EQ(view.size(), 2);
  double base = 0.0;
  for (const auto &item : view) {
    auto subview = item.get<DatasetView<const Data::Value>>();
    ASSERT_EQ(subview.size(), 3);
    auto it = subview.begin();
    EXPECT_EQ(it++->get<Data::Value>(), base + 1.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 3.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 5.0);
    base += 1.0;
  }
}

TEST(DatasetView, nested_DatasetView_all_subdimension_combinations_3D) {
  Dataset d;
  d.insert<Data::Value>(
      "name1", Dimensions({{Dim::Z, 2}, {Dim::Y, 3}, {Dim::X, 4}}),
      {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
       13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0});

  DatasetView<DatasetView<const Data::Value>> viewX(d, {Dim::Y, Dim::Z});
  ASSERT_EQ(viewX.size(), 4);
  double base = 0.0;
  for (const auto &item : viewX) {
    auto subview = item.get<DatasetView<const Data::Value>>();
    ASSERT_EQ(subview.size(), 6);
    auto it = subview.begin();
    EXPECT_EQ(it++->get<Data::Value>(), base + 1.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 5.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 9.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 13.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 17.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 21.0);
    base += 1.0;
  }

  DatasetView<DatasetView<const Data::Value>> viewY(d, {Dim::X, Dim::Z});
  ASSERT_EQ(viewY.size(), 3);
  base = 0.0;
  for (const auto &item : viewY) {
    auto subview = item.get<DatasetView<const Data::Value>>();
    ASSERT_EQ(subview.size(), 8);
    auto it = subview.begin();
    EXPECT_EQ(it++->get<Data::Value>(), base + 1.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 2.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 3.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 4.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 13.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 14.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 15.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 16.0);
    base += 4.0;
  }

  DatasetView<DatasetView<const Data::Value>> viewZ(d, {Dim::X, Dim::Y});
  ASSERT_EQ(viewZ.size(), 2);
  base = 0.0;
  for (const auto &item : viewZ) {
    auto subview = item.get<DatasetView<const Data::Value>>();
    ASSERT_EQ(subview.size(), 12);
    auto it = subview.begin();
    EXPECT_EQ(it++->get<Data::Value>(), base + 1.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 2.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 3.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 4.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 5.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 6.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 7.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 8.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 9.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 10.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 11.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 12.0);
    base += 12.0;
  }

  DatasetView<DatasetView<const Data::Value>> viewYZ(d, {Dim::X});
  ASSERT_EQ(viewYZ.size(), 6);
  base = 0.0;
  for (const auto &item : viewYZ) {
    auto subview = item.get<DatasetView<const Data::Value>>();
    ASSERT_EQ(subview.size(), 4);
    auto it = subview.begin();
    EXPECT_EQ(it++->get<Data::Value>(), base + 1.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 2.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 3.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 4.0);
    base += 4.0;
  }

  DatasetView<DatasetView<const Data::Value>> viewXZ(d, {Dim::Y});
  ASSERT_EQ(viewXZ.size(), 8);
  base = 0.0;
  for (const auto &item : viewXZ) {
    auto subview = item.get<DatasetView<const Data::Value>>();
    ASSERT_EQ(subview.size(), 3);
    auto it = subview.begin();
    EXPECT_EQ(it++->get<Data::Value>(), base + 1.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 5.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 9.0);
    base += 1.0;
    // Jump to next Z
    if (base == 4.0)
      base += 8.0;
  }

  DatasetView<DatasetView<const Data::Value>> viewXY(d, {Dim::Z});
  ASSERT_EQ(viewXY.size(), 12);
  base = 0.0;
  for (const auto &item : viewXY) {
    auto subview = item.get<DatasetView<const Data::Value>>();
    ASSERT_EQ(subview.size(), 2);
    auto it = subview.begin();
    EXPECT_EQ(it++->get<Data::Value>(), base + 1.0);
    EXPECT_EQ(it++->get<Data::Value>(), base + 13.0);
    base += 1.0;
  }
}

TEST(DatasetView, nested_DatasetView_constant_variable) {
  Dataset d;
  d.insert<Data::Value>("name1", Dimensions({{Dim::Z, 2}, {Dim::X, 4}}),
                        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  d.insert<Coord::X>({Dim::X, 4}, {10.0, 20.0, 30.0, 40.0});

  // Coord::X has fewer dimensions, throws if not const when not nested...
  EXPECT_THROW_MSG(
      (DatasetView<const Data::Value, Coord::X>(d)), std::runtime_error,
      "Variables requested for iteration have different dimensions");
  // ... and also when nested.
  EXPECT_THROW_MSG(
      (DatasetView<DatasetView<const Data::Value, Coord::X>>(d, {Dim::X})),
      std::runtime_error,
      "Variables requested for iteration have different dimensions");

  DatasetView<DatasetView<const Data::Value, const Coord::X>> view(d, {Dim::X});
  ASSERT_EQ(view.size(), 2);
  double value = 0.0;
  for (const auto &item : view) {
    auto subview = item.get<DatasetView<const Data::Value, const Coord::X>>();
    ASSERT_EQ(subview.size(), 4);
    double x = 0.0;
    for (const auto &subitem : subview) {
      x += 10.0;
      value += 1.0;
      EXPECT_EQ(subitem.get<Coord::X>(), x);
      EXPECT_EQ(subitem.get<Data::Value>(), value);
    }
  }
}

TEST(DatasetView, nested_DatasetView_copy_on_write) {
  Dataset d;
  d.insert<Data::Value>("name1", Dimensions({{Dim::Y, 2}, {Dim::X, 2}}),
                        {1.0, 2.0, 3.0, 4.0});
  d.insert<Coord::X>(Dimensions({{Dim::Y, 2}, {Dim::X, 2}}),
                     {10.0, 20.0, 30.0, 40.0});

  auto copy(d);

  DatasetView<DatasetView<const Data::Value, const Coord::X>> const_view(
      copy, {Dim::X});

  EXPECT_EQ(&d.get<const Data::Value>()[0],
            &(const_view.begin()
                  ->get<DatasetView<const Data::Value, const Coord::X>>()
                  .begin()
                  ->get<Data::Value>()));
  EXPECT_EQ(&d.get<const Coord::X>()[0],
            &(const_view.begin()
                  ->get<DatasetView<const Data::Value, const Coord::X>>()
                  .begin()
                  ->get<Coord::X>()));

  DatasetView<DatasetView<const Data::Value, Coord::X>> partially_const_view(
      copy, {Dim::X});

  EXPECT_EQ(&d.get<const Data::Value>()[0],
            &(partially_const_view.begin()
                  ->get<DatasetView<const Data::Value, Coord::X>>()
                  .begin()
                  ->get<Data::Value>()));
  EXPECT_NE(&d.get<const Coord::X>()[0],
            &(partially_const_view.begin()
                  ->get<DatasetView<const Data::Value, Coord::X>>()
                  .begin()
                  ->get<Coord::X>()));

  DatasetView<DatasetView<Data::Value, Coord::X>> nonconst_view(copy, {Dim::X});

  EXPECT_NE(&d.get<const Data::Value>()[0],
            &(nonconst_view.begin()
                  ->get<DatasetView<Data::Value, Coord::X>>()
                  .begin()
                  ->get<Data::Value>()));
  EXPECT_NE(&d.get<const Coord::X>()[0],
            &(nonconst_view.begin()
                  ->get<DatasetView<Data::Value, Coord::X>>()
                  .begin()
                  ->get<Coord::X>()));
}

TEST(DatasetView, histogram_using_nested_DatasetView) {
  Dataset d;
  // Edges do not have Dim::Spectrum, "shared" by all histograms.
  d.insert<Coord::Tof>(Dimensions(Dim::Tof, 3), {10.0, 20.0, 30.0});
  Dimensions dims;
  dims.add(Dim::Tof, 2);
  dims.add(Dim::Spectrum, 4);
  d.insert<Data::Value>("sample", dims,
                        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  d.insert<Data::Variance>("sample", dims, 8);
  d.insert<Coord::SpectrumNumber>({Dim::Spectrum, 4}, {1, 2, 3, 4});

  using HistogramView =
      DatasetView<Bin<Coord::Tof>, Data::Value, Data::Variance>;
  DatasetView<HistogramView, Coord::SpectrumNumber> view(d, {Dim::Tof});

  EXPECT_EQ(view.size(), 4);
  int32_t specNum = 1;
  double value = 1.0;
  for (const auto &item : view) {
    EXPECT_EQ(item.get<Coord::SpectrumNumber>(), specNum++);
    auto histview = item.get<HistogramView>();
    EXPECT_EQ(histview.size(), 2);
    double edge = 10.0;
    for (const auto &bin : histview) {
      EXPECT_EQ(bin.left(), edge);
      EXPECT_EQ(bin.right(), edge + 10.0);
      edge += 10.0;
      EXPECT_EQ(bin.value(), value++);
    }
  }

  auto it = view.begin();
  auto histogram = it->get<HistogramView>();
  EXPECT_EQ(histogram.size(), 2);
  auto bin = histogram.begin();
  EXPECT_EQ(bin->value(), 1.0);
  ++bin;
  EXPECT_EQ(bin->value(), 2.0);
  bin->value() += 0.2;
  EXPECT_EQ(d.get<Data::Value>()[1], 2.2);
  it++;
  EXPECT_EQ(it->get<HistogramView>().begin()->value(), 3.0);
}

TEST(DatasetView, single_column_edges) {
  Dataset d;
  d.insert<Coord::Tof>(Dimensions(Dim::Tof, 3), 3);
  d.insert<Data::Int>("name2", Dimensions(Dim::Tof, 2), 2);
  auto var = d.get<Coord::Tof>();
  ASSERT_EQ(var.size(), 3);
  var[0] = 0.2;
  var[2] = 2.2;

  DatasetView<Coord::Tof> view(d);
  auto it = view.begin();
  ASSERT_LT(it, view.end());
  ASSERT_EQ(it->get<Coord::Tof>(), 0.2);
  it++;
  ASSERT_LT(it, view.end());
  ASSERT_EQ(it->get<Coord::Tof>(), 0.0);
  ASSERT_LT(it, view.end());
  it++;
  ASSERT_EQ(it->get<Coord::Tof>(), 2.2);
  ASSERT_LT(it, view.end());
  it++;
  ASSERT_EQ(it, view.end());
}

TEST(DatasetView, single_column_bins) {
  Dataset d;
  d.insert<Coord::Tof>(Dimensions(Dim::Tof, 3), 3);
  d.insert<Data::Int>("name2", Dimensions(Dim::Tof, 2), 2);
  auto var = d.get<Coord::Tof>();
  ASSERT_EQ(var.size(), 3);
  var[0] = 0.2;
  var[1] = 1.2;
  var[2] = 2.2;

  DatasetView<Bin<Coord::Tof>> view(d);
  auto it = view.begin();
  it++;
  ASSERT_NE(it, view.end());
  it++;
  // Lenth of edges is 3, but there are only 2 bins!
  ASSERT_EQ(it, view.end());
}

TEST(DatasetView, multi_column_edges) {
  Dataset d;
  d.insert<Coord::Tof>(Dimensions(Dim::Tof, 3), 3);
  d.insert<Data::Int>("name2", Dimensions(Dim::Tof, 2), 2);
  auto var = d.get<Coord::Tof>();
  var[0] = 0.2;
  var[1] = 1.2;
  var[2] = 2.2;

  // Cannot simultaneously iterate edges and non-edges, so this throws.
  EXPECT_THROW_MSG((DatasetView<Coord::Tof, Data::Int>(d)), std::runtime_error,
                   "Variables requested for iteration do not span a joint "
                   "space. In case one of the variables represents bin edges "
                   "direct joint iteration is not possible. Use the Bin<> "
                   "wrapper to iterate over bins defined by edges instead.");

  DatasetView<Bin<Coord::Tof>, Data::Int> view(d);
  // TODO What are good names for named getters? tofCenter(), etc.?
  const auto &bin = view.begin()->get<Bin<Coord::Tof>>();
  EXPECT_EQ(bin.center(), 0.7);
  EXPECT_EQ(bin.width(), 1.0);
  EXPECT_EQ(bin.left(), 0.2);
  EXPECT_EQ(bin.right(), 1.2);
}

TEST(DatasetView, multi_dimensional_edges) {
  Dataset d;
  d.insert<Coord::X>(Dimensions({{Dim::Y, 2}, {Dim::X, 3}}),
                     {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  // TODO There is currently a bug in DatasetView: If `Bin` iteration is
  // requested but the dataset contains only edges the shape calculation gives
  // wrong results.
  d.insert<Data::Value>("", {Dim::X, 2});

  DatasetView<Bin<Coord::X>> view(d);
  ASSERT_EQ(view.size(), 4);
  auto it = view.begin();
  EXPECT_EQ(it++->get<Bin<Coord::X>>().left(), 1.0);
  EXPECT_EQ(it++->get<Bin<Coord::X>>().left(), 2.0);
  EXPECT_EQ(it++->get<Bin<Coord::X>>().left(), 4.0);
  EXPECT_EQ(it++->get<Bin<Coord::X>>().left(), 5.0);
  it -= 4;
  EXPECT_EQ(it++->get<Bin<Coord::X>>().right(), 2.0);
  EXPECT_EQ(it++->get<Bin<Coord::X>>().right(), 3.0);
  EXPECT_EQ(it++->get<Bin<Coord::X>>().right(), 5.0);
  EXPECT_EQ(it++->get<Bin<Coord::X>>().right(), 6.0);
}

TEST(DatasetView, edges_are_not_inner_dimension) {
  Dataset d;
  d.insert<Coord::Y>(Dimensions({{Dim::Y, 2}, {Dim::X, 3}}),
                     {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  d.insert<Data::Value>("", {Dim::Y, 1});

  DatasetView<Bin<Coord::Y>> view(d);
  ASSERT_EQ(view.size(), 3);
  auto it = view.begin();
  EXPECT_EQ(it++->get<Bin<Coord::Y>>().left(), 1.0);
  EXPECT_EQ(it++->get<Bin<Coord::Y>>().left(), 2.0);
  EXPECT_EQ(it++->get<Bin<Coord::Y>>().left(), 3.0);
  it -= 3;
  EXPECT_EQ(it++->get<Bin<Coord::Y>>().right(), 4.0);
  EXPECT_EQ(it++->get<Bin<Coord::Y>>().right(), 5.0);
  EXPECT_EQ(it++->get<Bin<Coord::Y>>().right(), 6.0);
}

TEST(DatasetView, named_getter) {
  Dataset d;
  d.insert<Coord::Tof>(Dimensions(Dim::Tof, 3), 3);
  auto var = d.get<Coord::Tof>();
  ASSERT_EQ(var.size(), 3);
  var[0] = 0.2;
  var[2] = 2.2;

  DatasetView<Coord::Tof> view(d);
  auto it = view.begin();
  ASSERT_EQ(it->tof(), 0.2);
  it++;
  ASSERT_EQ(it->tof(), 0.0);
  it++;
  ASSERT_EQ(it->tof(), 2.2);
}

TEST(DatasetView, duplicate_data_tag) {
  Dataset d;
  d.insert<Data::Value>("name1", Dimensions{}, 1);
  d.insert<Data::Value>("name2", Dimensions{}, 1);

  EXPECT_THROW_MSG(DatasetView<Data::Value> view(d), std::runtime_error,
                   "Given variable tag is not unique. Must provide a name.");
  EXPECT_NO_THROW(DatasetView<Data::Value> view(d, "name2"));
}

TEST(DatasetView, named_variable_and_coordinate) {
  Dataset d;
  d.insert<Coord::X>(Dimensions{}, 1);
  d.insert<Data::Value>("name", Dimensions{}, 1);

  EXPECT_NO_THROW((DatasetView<Coord::X, Data::Value>(d, "name")));
  (DatasetView<Coord::X, Data::Value>(d, "name"));
}

TEST(DatasetView, spectrum_position) {
  Dataset d;
  d.insert<Coord::DetectorPosition>({Dim::Detector, 4}, {1.0, 2.0, 4.0, 8.0});
  Vector<boost::container::small_vector<gsl::index, 1>> grouping = {
      {0, 2}, {1}, {}};
  d.insert<Coord::DetectorGrouping>({Dim::Spectrum, 3}, grouping);

  DatasetView<Coord::SpectrumPosition> view(d);
  auto it = view.begin();
  EXPECT_EQ(it->get<Coord::SpectrumPosition>(), 2.5);
  ++it;
  EXPECT_EQ(it->get<Coord::SpectrumPosition>(), 2.0);
  ++it;
  EXPECT_THROW_MSG(it->get<Coord::SpectrumPosition>(), std::runtime_error,
                   "Spectrum has no detectors, cannot get position.");
  ++it;
  ASSERT_EQ(it, view.end());
}

TEST(DatasetView, derived_standard_deviation) {
  Dataset d;
  d.insert<Data::Variance>("data", {Dim::X, 3}, {4.0, 9.0, -1.0});
  DatasetView<Data::StdDev> view(d);
  auto it = view.begin();
  EXPECT_EQ(it->get<Data::StdDev>(), 2.0);
  ++it;
  EXPECT_EQ(it->get<Data::StdDev>(), 3.0);
  ++it;
  EXPECT_TRUE(std::isnan(it->get<Data::StdDev>()));
}

TEST(DatasetView, type_sorting) {
  Dataset data;
  data.insert<Coord::X>({}, 1);
  data.insert<Coord::Y>({}, 1);
  DatasetView<Coord::X, Coord::Y> a(data);
  DatasetView<Coord::Y, Coord::X> b(data);
  DatasetView<Coord::Y, const Coord::X> b_const(data);
  EXPECT_EQ(typeid(decltype(a)), typeid(decltype(b)));
  EXPECT_NE(typeid(decltype(a)), typeid(decltype(b_const)));
}

TEST(DatasetView, type_sorting_nested) {
  Dataset data;
  data.insert<Coord::X>({}, 1);
  data.insert<Coord::Y>({}, 1);
  DatasetView<Coord::X, DatasetView<Coord::Y>> a(data);
  DatasetView<DatasetView<Coord::Y>, Coord::X> b(data);
  EXPECT_EQ(typeid(decltype(a)),
            typeid(DatasetViewImpl<Coord::X, DatasetViewImpl<Coord::Y>>));
  EXPECT_EQ(typeid(decltype(a)), typeid(decltype(b)));
}

TEST(DatasetView, type_sorting_two_nested) {
  Dataset data;
  data.insert<Coord::X>({}, 1);
  data.insert<Coord::Y>({}, 1);
  data.insert<Coord::Z>({}, 1);
  DatasetView<Coord::X, DatasetView<Coord::Y, Coord::Z>> a(data);
  DatasetView<Coord::X, DatasetView<Coord::Z, Coord::Y>> b(data);
  DatasetView<DatasetView<Coord::Y, Coord::Z>, Coord::X> c(data);
  DatasetView<DatasetView<Coord::Z, Coord::Y>, Coord::X> d(data);
  EXPECT_EQ(
      typeid(decltype(a)),
      typeid(DatasetViewImpl<Coord::X, DatasetViewImpl<Coord::Y, Coord::Z>>));
  EXPECT_EQ(typeid(decltype(a)), typeid(decltype(b)));
  EXPECT_EQ(typeid(decltype(a)), typeid(decltype(c)));
  EXPECT_EQ(typeid(decltype(a)), typeid(decltype(d)));
  DatasetView<Coord::X, DatasetView<const Coord::Y, Coord::Z>> a_const(data);
  EXPECT_EQ(typeid(decltype(a_const)),
            typeid(DatasetViewImpl<Coord::X,
                                   DatasetViewImpl<const Coord::Y, Coord::Z>>));
}
