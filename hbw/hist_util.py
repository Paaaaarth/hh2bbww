# coding: utf-8

"""
Task to produce and merge histograms.
"""

from __future__ import annotations

import law
import order as od

from columnflow.util import maybe_import

hist = maybe_import("hist")
np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


def add_axis(histogram: hist.Hist, variable_inst: od.Variable) -> hist.Hist:
    """
    Add an axis to a histogram based on a variable instance. The axis_type is chosen
    based on the variable instance's "axis_type" auxiliary.

    :param histogram: The histogram to add the axis to.
    :param variable_inst: The variable instance to use for the axis.
    :return: The histogram with the added axis.
    """
    default_kwargs = {
        "name": variable_inst.name,
        "label": variable_inst.get_full_x_title(),
    }

    axis_kwargs = law.util.merge_dicts(
        default_kwargs,
        variable_inst.x("axis_kwargs", {}),
        deep=True,
    )

    default_axis_type = "integer" if variable_inst.discrete_x else "variable"
    axis_type = variable_inst.x("axis_type", default_axis_type).lower()

    if axis_type == "variable" or axis_type == "var":
        return histogram.Var(
            variable_inst.bin_edges,
            **axis_kwargs,
        )
    elif axis_type == "integer" or axis_type == "int":
        return histogram.Integer(
            int(variable_inst.bin_edges[0]),
            int(variable_inst.bin_edges[-1]),
            **axis_kwargs,
        )
    elif axis_type == "boolean" or axis_type == "bool":
        return histogram.Boolean(
            **axis_kwargs,
        )
    elif axis_type == "intcategory" or axis_type == "intcat":
        binning = [int(b) for b in variable_inst.binning] if isinstance(variable_inst.binning, list) else []
        return histogram.IntCat(
            binning,
            growth=True,
            **axis_kwargs,
        )
    elif axis_type == "strcategory" or axis_type == "strcat":
        return histogram.StrCat(
            [],
            growth=True,
            **axis_kwargs,
        )
    elif axis_type == "regular" or axis_type == "reg":
        return histogram.Regular(
            variable_inst.nbins,
            variable_inst.bin_edges[0],
            variable_inst.bin_edges[-1],
            **axis_kwargs,
        )


def create_columnflow_hist(
    *variable_insts,
    add_default_axes: bool = False,
) -> hist.Hist:
    histogram = hist.Hist.new

    # default axes
    if add_default_axes:
        histogram = histogram.IntCat([], name="category", growth=True)
        histogram = histogram.IntCat([], name="process", growth=True)
        histogram = histogram.IntCat([], name="shift", growth=True)

    # requested axes
    for variable_inst in variable_insts:
        histogram = add_axis(histogram, variable_inst)

    histogram = histogram.Weight()

    return histogram
