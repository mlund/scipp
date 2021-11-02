from ._scipp.core import Variable


def _wrap_numpy(func, *args, **kwargs):
    # if func.__name__ in self.special_functions:
    #     unit = func(self.unit, *args[1:], **kwargs)
    # else:
    #     unit = self.unit
    if isinstance(args[0], tuple) or isinstance(args[0], list):
        # # Case where we have a sequence of arrays, e.g. `concatenate`
        # for a in args[0]:
        #     if a.unit != unit:
        #         self._raise_incompatible_units_error(a, func.__name__)
        args = (tuple(a.values for a in args[0]), ) + args[1:]
    # elif (len(args) > 1 and hasattr(args[1], "_array")):
    #     if hasattr(args[0], "_array"):
    #         # Case of a binary operation, with two Arrays, e.g. `dot`
    #         # TODO: what should we do with the unit? Apply the func to it?
    #         # unit = func(args[0].unit, args[1].unit, *args[2:], **kwargs)
    #         args = (args[0]._array, args[1]._array) + args[2:]
    #     else:
    #         # Case of a binary operation: ndarray with Array
    #         # In this case, only multiply is allowed?
    #         if func.__name__ != "multiply":
    #             raise RuntimeError("Cannot use operation {} between ndarray and "
    #                                "Array".format(func.__name__))
    #         args = (args[0], args[1]._array) + args[2:]
    else:
        args = (args[0].values, ) + args[1:]
    return func(*args, **kwargs)
    # return result


def _array_ufunc(var, ufunc, method, *inputs, **kwargs):
    """
    Numpy array_ufunc protocol to allow Array to work with numpy ufuncs.
    """
    if method != "__call__":
        # Only handle ufuncs as callables
        return NotImplemented
    result = _wrap_numpy(ufunc, *inputs, **kwargs)
    # out.unit = var.unit
    return Variable(dims=var.dims, values=result, unit=var.unit)


def _array_function(var, func, types, args, kwargs):
    """
    Numpy array_function protocol to allow Array to work with numpy
    functions.
    """
    result = _wrap_numpy(func, *args, **kwargs)
    return Variable(dims=var.dims, values=result, unit=var.unit)
