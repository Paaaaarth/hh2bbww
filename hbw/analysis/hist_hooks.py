# coding: utf-8

"""
Histogram hooks.
"""

from __future__ import annotations

from functools import partial

import law
import order as od

from columnflow.util import maybe_import, DotDict
from hbw.hist_util import apply_rebinning_edges
from columnflow.types import TYPE_CHECKING

np = maybe_import("numpy")
if TYPE_CHECKING:
    hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


def cumsum(
    task,
    hists: hist.Histogram,
    reverse: bool = False,
    **kwargs,
):
    for config_inst, proc_hists in hists.items():
        for proc_inst, proc_hist in proc_hists.items():
            if reverse:
                proc_hist.values()[...] = np.cumsum(proc_hist.values()[..., ::-1], axis=-1)[..., ::-1]
            else:
                proc_hist.values()[...] = np.cumsum(proc_hist.values(), axis=-1)

    return hists


def rebin(task, hists: hist.Histogram, **kwargs):
    """
    Rebin histograms with edges that are pre-defined for a certain variable and category.
    Lots of hard-coded stuff at the moment.
    """
    # get variable inst assuming we created a 1D histogram
    variable_inst = task.config_inst.get_variable(task.branch_data.variable)
    category_inst = task.config_inst.get_category(task.branch_data.category)
    edges_path = "/data/dust/user/patilpar/test_repo/hh2bbww/data/cf_store/hbw_dl/hbw.ModifyDatacardsFlatRebin/c22prev14__c22postv14__c23prev14__c23postv14/calib__ak4V5__ak8V5__eleV6/sel__dl1V0/red__default/prod__event_weightsV5__dl_ml_inputsV3__cats_ml_dl_combo_multiV5/hist__with_trigger_weightV2/inf__dl_combo_multi_binaryV15/ml__dl_combo_multi__405bcdc426__dl_combo_binary__b400366eea/iter2"  # noqa: E501

    edges_filename = {
        "sr__2b__ml_sig_all": "edges_10__cfg_2022_2023__cat_sr__2b__ml_sig_all.json",
        "sr__3b__ml_sig_all": "edges_10__cfg_2022_2023__cat_sr__3b__ml_sig_all.json",   
        "sr__4b__ml_sig_all": "edges_10__cfg_2022_2023__cat_sr__4b__ml_sig_all.json",
    }[category_inst.name]
    import json
    with open(f"{edges_path}/{edges_filename}", "r") as f:
        edges = json.load(f)

    h_rebinned = DotDict()
    for config_inst, proc_hists in hists.items():
        h_rebinned[config_inst] = DotDict()
        for proc_inst, proc_hist in proc_hists.items():
            old_axis = proc_hist.axes[variable_inst.name]

            h_rebin = apply_rebinning_edges(proc_hist.copy(), old_axis.name, edges)

            if not np.isclose(proc_hist.sum().value, h_rebin.sum().value):
                raise Exception(f"Rebinning changed histogram value: {proc_hist.sum().value} -> {h_rebin.sum().value}")
            if not np.isclose(proc_hist.sum().variance, h_rebin.sum().variance):
                raise Exception(f"Rebinning changed histogram variance: {proc_hist.sum().variance} -> {h_rebin.sum().variance}")  # noqa: E501
            h_rebinned[config_inst][proc_inst] = h_rebin

    return h_rebinned


def blind_bins_above_score(task, hists: hist.Histogram, default_cut=0.8, **kwargs):
    var_name = task.branch_data.variable
    if "logit" in var_name:
        # identify logit transformed scores and convert cut accordingly
        score_cut = np.log(default_cut / (1 - default_cut))
    elif "mlscore.sig_" in var_name:
        # only apply blinding for signal scores
        score_cut = default_cut
    else:
        # do nothing
        return hists

    for config_inst, proc_hists in hists.items():
        for proc_inst, proc_hist in proc_hists.items():
            if proc_inst.is_data:
                # blind data above score cut
                axis = proc_hist.axes[var_name]
                bin_indices = np.where(axis.edges[1:] > score_cut)[0]
                proc_hist.values()[..., bin_indices] = -1
                proc_hist.variances()[..., bin_indices] = 0

    return hists


def blind_bins(task, hists: hist.Histogram, blinding_threshold=0.08, **kwargs):
    from columnflow.plotting.plot_util import blind_sensitive_bins

    out_hists = {}
    for config_inst, hists in hists.items():
        # unify histogram shapes
        hist_list = list(hists.values())
        zero_hist = sum([h * 0 for h in hist_list[1:]], hist_list[0] * 0)
        hists = {proc: zero_hist + h for proc, h in hists.items()}

        # apply blinding if s/sqrt(b) > blinding_threshold
        # NOTE: this does not yet work for Multi-dim histograms that include categories as is the case here....
        out_hists[config_inst] = blind_sensitive_bins(
            hists, config_inst, threshold=blinding_threshold, remove_mc=True,
        )
    return out_hists


def add_hist_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to an analysis.
    """
    # add hist hooks to analysis instance
    analysis_inst.x.hist_hooks = {
        "cumsum": cumsum,
        "cumsum_reverse": partial(cumsum, reverse=True),
        "rebin": rebin,
        # "blind": blind_bins,
        "blind_bins_above_score": blind_bins_above_score,
    }
