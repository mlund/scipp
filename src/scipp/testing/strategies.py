# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
from functools import partial
from typing import Optional, Sequence, Union

from hypothesis.errors import InvalidArgument
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from ..core import variable as creation
from ..core import DataArray, DType


def dims() -> st.SearchStrategy:
    # Allowing all graphic utf-8 characters and control characters
    # except NULL, which causes problems in C and C++ code (e.g. HDF5).
    return st.text(st.characters(
        whitelist_categories=['L', 'M', 'N', 'P', 'S', 'Zs', 'Cc'],
        blacklist_characters='\0'),
                   min_size=1,
                   max_size=10)


def sizes_dicts(
        ndim: Optional[Union[int, st.SearchStrategy]] = None) -> st.SearchStrategy:
    if isinstance(ndim, st.SearchStrategy):
        return ndim.flatmap(lambda n: sizes_dicts(ndim=n))
    keys = dims()
    values = st.integers(min_value=1, max_value=10)
    if ndim is None:
        # The constructor of sc.Variable in Python only supports
        # arrays with <= 4 dimensions.
        return st.dictionaries(keys=keys, values=values, min_size=0, max_size=4)
    return st.dictionaries(keys=keys, values=values, min_size=ndim, max_size=ndim)


def units() -> st.SearchStrategy:
    return st.sampled_from(('one', 'm', 'kg', 's', 'A', 'K', 'count'))


def integer_dtypes(sizes: Sequence[int] = (32, 64)) -> st.SearchStrategy:
    return st.sampled_from([f'int{size}' for size in sizes])


def floating_dtypes(sizes: Sequence[int] = (32, 64)) -> st.SearchStrategy:
    return st.sampled_from([f'float{size}' for size in sizes])


def scalar_numeric_dtypes() -> st.SearchStrategy:
    return st.sampled_from((integer_dtypes, floating_dtypes)).flatmap(lambda f: f())


@st.composite
def fixed_variables(draw, dtype, sizes) -> st.SearchStrategy:
    values = draw(npst.arrays(dtype, shape=tuple(sizes.values())))
    if dtype == float and draw(st.booleans()):
        variances = draw(npst.arrays(dtype, shape=values.shape))
    else:
        variances = None
    return creation.array(dims=list(sizes.keys()),
                          values=values,
                          variances=variances,
                          unit=draw(units()))


@st.composite
def _make_vectors(draw, sizes):
    values = draw(npst.arrays(float, (*sizes.values(), 3)))
    return creation.vectors(dims=tuple(sizes), values=values, unit=draw(units()))


@st.composite
def vectors(draw, ndim=None) -> st.SearchStrategy:
    if ndim is None:
        ndim = draw(st.integers(0, 3))
    return draw(sizes_dicts(ndim).flatmap(lambda s: _make_vectors(s)))


def use_variances(dtype) -> st.SearchStrategy:
    if dtype in (DType.float32, DType.float64):
        return st.booleans()
    return st.just(False)


def _variables_from_fixed_args(args) -> st.SearchStrategy:

    def make_array():
        return npst.arrays(args['dtype'], tuple(args['sizes'].values()))

    return st.builds(partial(creation.array,
                             dims=list(args['sizes'].keys()),
                             unit=args['unit']),
                     values=make_array(),
                     variances=make_array() if args['with_variances'] else st.none())


@st.composite
def variable_args(draw,
                  *,
                  ndim=None,
                  sizes=None,
                  unit=None,
                  dtype=None,
                  with_variances=None) -> dict:
    if ndim is not None:
        if sizes is not None:
            raise InvalidArgument('Arguments `ndim` and `sizes` cannot both be used. '
                                  f'Got ndim={ndim}, sizes={sizes}.')
    if sizes is None:
        sizes = sizes_dicts(ndim)
    if isinstance(sizes, st.SearchStrategy):
        sizes = draw(sizes)

    if unit is None:
        unit = units()
    if isinstance(unit, st.SearchStrategy):
        unit = draw(unit)

    if dtype is None:
        # TODO other dtypes?
        dtype = scalar_numeric_dtypes()
    if isinstance(dtype, st.SearchStrategy):
        dtype = draw(dtype)

    if with_variances is None:
        with_variances = use_variances(dtype)
    if isinstance(with_variances, st.SearchStrategy):
        with_variances = draw(with_variances)

    return dict(sizes=sizes, unit=unit, dtype=dtype, with_variances=with_variances)


def variables(*,
              ndim=None,
              sizes=None,
              unit=None,
              dtype=None,
              with_variances=None) -> st.SearchStrategy:
    return variable_args(ndim=ndim,
                         sizes=sizes,
                         unit=unit,
                         dtype=dtype,
                         with_variances=with_variances).flatmap(
                             lambda args: _variables_from_fixed_args(args))


def n_variables(n: int,
                *,
                ndim=None,
                sizes=None,
                unit=None,
                dtype=None,
                with_variances=None) -> st.SearchStrategy:
    return variable_args(ndim=ndim,
                         sizes=sizes,
                         unit=unit,
                         dtype=dtype,
                         with_variances=with_variances).flatmap(lambda args: st.tuples(
                             *(_variables_from_fixed_args(args) for _ in range(n))))


@st.composite
def coord_dicts_1d(draw, sizes) -> st.SearchStrategy:
    return {
        dim: draw(fixed_variables(dtype=int, sizes={dim: size}))
        for dim, size in sizes.items()
    }


@st.composite
def dataarrays(draw) -> st.SearchStrategy:
    data = draw(variables(dtype=float))
    coords = draw(coord_dicts_1d(sizes=data.sizes))
    return DataArray(data, coords=coords)
