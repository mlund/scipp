// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/dataset/map_view.h"
#include "scipp/variable/arithmetic.h"
#include "scipp/variable/reduction.h"

#include "../variable/operations_common.h"
#include "dataset_operations_common.h"

namespace scipp::dataset {

/// Flatten with mask, skipping masked elements.
Variable flatten(const VariableConstView &var, const Dim dim,
                 const MasksConstView &masks) {
  auto dims = var.dims();
  dims.erase(dim);
  Variable flattened(var, dims);
  auto mask = irreducible_mask(masks, dim);
  if (mask)
    mask = ~std::move(mask);
  else
    mask = makeVariable<bool>(Values{true});
  flatten_impl(flattened, var, mask);
  return flattened;
}

Variable sum(const VariableConstView &var, const Dim dim,
             const MasksConstView &masks) {
  if (const auto mask_union = irreducible_mask(masks, dim))
    return sum(var * ~mask_union, dim);
  return sum(var, dim);
}

VariableView sum(const VariableConstView &var, const Dim dim,
                 const MasksConstView &masks, const VariableView &out) {
  if (const auto mask_union = irreducible_mask(masks, dim))
    return sum(var * ~mask_union, dim, out);
  return sum(var, dim, out);
}

Variable mean(const VariableConstView &var, const Dim dim,
              const MasksConstView &masks) {
  if (const auto mask_union = irreducible_mask(masks, dim)) {
    const auto masks_sum = sum(mask_union, dim);
    return mean_impl(var * ~mask_union, dim, masks_sum);
  }
  return mean(var, dim);
}

VariableView mean(const VariableConstView &var, const Dim dim,
                  const MasksConstView &masks, const VariableView &out) {
  if (const auto mask_union = irreducible_mask(masks, dim)) {
    const auto masks_sum = sum(mask_union, dim);
    return mean_impl(var * ~mask_union, dim, masks_sum, out);
  }
  return mean(var, dim, out);
}

/// Returns the union of all masks with irreducible dimension `dim`.
///
/// Irreducible means that a reduction operation must apply these masks since
/// depend on the reduction dimension. Returns an invalid (empty) variable if
/// there is no irreducible mask.
Variable irreducible_mask(const MasksConstView &masks, const Dim dim) {
  Variable mask_union;
  for (const auto &mask : masks) {
    if (mask.second.dims().contains(dim)) {
      mask_union =
          (mask_union ? mask_union : makeVariable<bool>(Values{false})) |
          mask.second;
    }
  }
  return mask_union;
}

/// Merges all the masks that have all their dimensions found in the given set
//  of dimensions.
Variable masks_merge_if_contained(const MasksConstView &masks,
                                  const Dimensions &dims) {
  auto mask_union = makeVariable<bool>(Values{false});
  for (const auto &mask : masks) {
    if (dims.contains(mask.second.dims()))
      mask_union = mask_union | mask.second;
  }
  return mask_union;
}

} // namespace scipp::dataset
