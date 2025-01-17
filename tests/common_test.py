# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest
import scipp as sc


def test_bool_raises():
    # Truth values of arrays are undefined
    var = sc.scalar(True)
    with pytest.raises(RuntimeError):
        var and var
    da = sc.DataArray(var)
    with pytest.raises(RuntimeError):
        da and da
    ds = sc.Dataset(data={'a': da})
    with pytest.raises(RuntimeError):
        ds and ds
