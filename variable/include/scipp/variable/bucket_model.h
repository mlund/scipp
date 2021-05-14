// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once
#include <algorithm>

#include "scipp/core/bucket_array_view.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/except.h"
#include "scipp/variable/arithmetic.h"
#include "scipp/variable/cumulative.h"
#include "scipp/variable/data_model.h"
#include "scipp/variable/except.h"
#include "scipp/variable/reduction.h"
#include "scipp/variable/util.h"

namespace scipp::variable {

template <class Indices> class BinModelBase : public VariableConcept {
public:
  BinModelBase(const VariableConceptHandle &indices, const Dim dim)
      : VariableConcept(units::one), m_indices(indices), m_dim(dim) {}

  scipp::index size() const override { return indices()->size(); }

  void setUnit(const units::Unit &unit) override {
    if (unit != units::one)
      throw except::UnitError(
          "Bins cannot have a unit. Did you mean to set the unit of the bin "
          "elements? This can be set, e.g., with `array.events.unit = 'm'`.");
  }

  bool hasVariances() const noexcept override { return false; }
  void setVariances(const Variable &) override {
    throw except::VariancesError("This data type cannot have variances.");
  }
  const Indices &bin_indices() const override { return indices(); }

  const auto &indices() const { return m_indices; }
  auto &indices() { return m_indices; }
  Dim bin_dim() const noexcept { return m_dim; }

private:
  Indices m_indices;
  Dim m_dim;
};

namespace {
template <class T> auto clone_impl(const ElementArrayModel<bucket<T>> &model) {
  return std::make_unique<ElementArrayModel<bucket<T>>>(
      model.indices()->clone(), model.bin_dim(), copy(model.buffer()));
}
} // namespace

SCIPP_VARIABLE_EXPORT void
expect_valid_bin_indices(const VariableConceptHandle &indices, const Dim dim,
                         const Sizes &buffer_sizes);

/// Specialization of ElementArrayModel for "binned" data. T could be Variable,
/// DataArray, or Dataset.
///
/// A bin in this context is defined as an element of a variable mapping to a
/// range of data, such as a slice of a DataArray.
template <class T>
class ElementArrayModel<bucket<T>>
    : public BinModelBase<VariableConceptHandle> {
  using Indices = VariableConceptHandle;

public:
  using value_type = bucket<T>;
  using range_type = typename bucket<T>::range_type;

  ElementArrayModel(const VariableConceptHandle &indices, const Dim dim,
                    T buffer)
      : BinModelBase<Indices>(indices, dim), m_buffer(std::move(buffer)) {}

  [[nodiscard]] VariableConceptHandle clone() const override {
    return clone_impl(*this);
  }

  bool operator==(const ElementArrayModel &other) const noexcept {
    const auto &i1 = requireT<const ElementArrayModel<range_type>>(*indices());
    const auto &i2 =
        requireT<const ElementArrayModel<range_type>>(*other.indices());
    return equals_impl(i1.values(), i2.values()) &&
           this->bin_dim() == other.bin_dim() && m_buffer == other.m_buffer;
  }
  bool operator!=(const ElementArrayModel &other) const noexcept {
    return !(*this == other);
  }

  [[nodiscard]] VariableConceptHandle
  makeDefaultFromParent(const scipp::index size) const override {
    return std::make_unique<ElementArrayModel>(
        makeVariable<range_type>(Dims{Dim::X}, Shape{size}).data_handle(),
        this->bin_dim(), T{m_buffer.slice({this->bin_dim(), 0, 0})});
  }

  [[nodiscard]] VariableConceptHandle
  makeDefaultFromParent(const Variable &shape) const override {
    const auto end = cumsum(shape);
    const auto begin = end - shape;
    const auto size = end.dims().volume() > 0
                          ? end.values<scipp::index>().as_span().back()
                          : 0;
    return std::make_unique<ElementArrayModel>(
        zip(begin, begin).data_handle(), this->bin_dim(),
        resize_default_init(m_buffer, this->bin_dim(), size));
  }

  static DType static_dtype() noexcept { return scipp::dtype<bucket<T>>; }
  [[nodiscard]] DType dtype() const noexcept override {
    return scipp::dtype<bucket<T>>;
  }

  [[nodiscard]] bool equals(const Variable &a,
                            const Variable &b) const override;
  void copy(const Variable &src, Variable &dest) const override;
  void copy(const Variable &src, Variable &&dest) const override;
  void assign(const VariableConcept &other) override;

  // TODO Should the mutable version return a view to prevent risk of clients
  // breaking invariants of variable?
  const T &buffer() const noexcept { return m_buffer; }
  T &buffer() noexcept { return m_buffer; }

  ElementArrayView<bucket<T>> values(const core::ElementArrayViewParams &base) {
    return {index_values(base), this->bin_dim(), m_buffer};
  }
  ElementArrayView<const bucket<T>>
  values(const core::ElementArrayViewParams &base) const {
    return {index_values(base), this->bin_dim(), m_buffer};
  }

  [[nodiscard]] scipp::index dtype_size() const override {
    return sizeof(range_type);
  }

private:
  auto index_values(const core::ElementArrayViewParams &base) const {
    return requireT<const ElementArrayModel<range_type>>(*this->indices())
        .values(base);
  }
  T m_buffer;
};

template <class T> using BinArrayModel = ElementArrayModel<core::bin<T>>;

template <class T>
bool ElementArrayModel<bucket<T>>::equals(const Variable &a,
                                          const Variable &b) const {
  // TODO This implementation is slow since it creates a view for every bucket.
  return equals_impl(a.values<bucket<T>>(), b.values<bucket<T>>());
}

template <class T>
void ElementArrayModel<bucket<T>>::copy(const Variable &src,
                                        Variable &dest) const {
  const auto &[indices0, dim0, buffer0] = src.constituents<bucket<T>>();
  auto &&[indices1, dim1, buffer1] = dest.constituents<bucket<T>>();
  static_cast<void>(dim1);
  copy_slices(buffer0, buffer1, dim0, indices0, indices1);
}
template <class T>
void ElementArrayModel<bucket<T>>::copy(const Variable &src,
                                        Variable &&dest) const {
  copy(src, dest);
}

template <class T>
void ElementArrayModel<bucket<T>>::assign(const VariableConcept &other) {
  *this = requireT<const ElementArrayModel<bucket<T>>>(other);
}

} // namespace scipp::variable
