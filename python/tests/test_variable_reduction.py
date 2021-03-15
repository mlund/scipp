# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @file
# @author Simon Heybrock
import scipp as sc
import numpy as np
from .common import assert_export


def test_all():
    var = sc.Variable(['x', 'y'],
                      values=np.array([True, True, True, False]).reshape(2, 2))
    assert sc.identical(sc.all(var), sc.Variable(value=False))


def test_all_with_dim():
    var = sc.Variable(['x', 'y'],
                      values=np.array([True, True, True, False]).reshape(2, 2))
    assert sc.identical(sc.all(var, 'x'),
                        sc.Variable(dims=['y'], values=[True, False]))
    assert sc.identical(sc.all(var, 'y'),
                        sc.Variable(dims=['x'], values=[True, False]))


def test_any():
    var = sc.Variable(['x', 'y'],
                      values=np.array([True, True, True, False]).reshape(2, 2))
    assert sc.identical(sc.any(var), sc.Variable(value=True))


def test_any_with_dim():
    var = sc.Variable(['x', 'y'],
                      values=np.array([True, True, False,
                                       False]).reshape(2, 2))
    assert sc.identical(sc.any(var, 'x'),
                        sc.Variable(dims=['y'], values=[True, True]))
    assert sc.identical(sc.any(var, 'y'),
                        sc.Variable(dims=['x'], values=[True, False]))


def test_min():
    var = sc.Variable(['x'], values=[1.0, 2.0, 3.0])
    assert sc.identical(sc.min(var, 'x'), sc.Variable(value=1.0))


def test_max():
    var = sc.Variable(['x'], values=[1.0, 2.0, 3.0])
    assert sc.identical(sc.max(var, 'x'), sc.Variable(value=3.0))


def test_nanmin():
    var = sc.Variable(['x'], values=np.array([1]))
    assert_export(sc.nanmin, var)
    assert_export(sc.nanmin, var, 'x')


def test_nanmax():
    var = sc.Variable(['x'], values=np.array([1]))
    assert_export(sc.nanmax, var)
    assert_export(sc.nanmax, var, 'x')


def test_sum():
    var = sc.Variable(['x', 'y'], values=np.arange(4.0).reshape(2, 2))
    assert sc.identical(sc.sum(var), sc.Variable(value=6.0))
    assert sc.identical(sc.sum(var, 'x'),
                        sc.Variable(dims=['y'], values=[2.0, 4.0]))
    out = sc.Variable(dims=['y'], values=np.zeros(2), dtype=sc.dtype.float64)
    sc.sum(var, 'x', out)
    assert sc.identical(out, sc.Variable(dims=['y'], values=[2.0, 4.0]))


def test_nansum():
    var = sc.Variable(['x', 'y'],
                      values=np.array([1.0, 1.0, 1.0, np.nan]).reshape(2, 2))
    assert sc.identical(sc.nansum(var), sc.Variable(value=3.0))
    assert sc.identical(sc.nansum(var, 'x'),
                        sc.Variable(dims=['y'], values=[2.0, 1.0]))
    out = sc.Variable(dims=['y'], values=np.zeros(2), dtype=sc.dtype.float64)
    sc.nansum(var, 'x', out)
    assert sc.identical(out, sc.Variable(dims=['y'], values=[2.0, 1.0]))


def test_mean():
    var = sc.Variable(['x', 'y'], values=np.arange(4.0).reshape(2, 2))
    assert sc.identical(sc.mean(var), sc.Variable(value=6.0 / 4))
    assert sc.identical(sc.mean(var, 'x'),
                        sc.Variable(dims=['y'], values=[1.0, 2.0]))
    out = sc.Variable(dims=['y'], values=np.zeros(2), dtype=sc.dtype.float64)
    sc.mean(var, 'x', out)
    assert sc.identical(out, sc.Variable(dims=['y'], values=[1.0, 2.0]))


def test_nanmean():
    var = sc.Variable(['x', 'y'],
                      values=np.array([1.0, 1.0, 1.0, 1.0]).reshape(2, 2))
    assert sc.identical(sc.nanmean(var), sc.Variable(value=3.0 / 3))
    assert sc.identical(sc.nanmean(var, 'x'),
                        sc.Variable(dims=['y'], values=[1.0, 1.0]))
    out = sc.Variable(dims=['y'], values=np.zeros(2), dtype=sc.dtype.float64)
    sc.mean(var, 'x', out)
    assert sc.identical(out, sc.Variable(dims=['y'], values=[1.0, 1.0]))
