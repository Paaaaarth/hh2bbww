# coding: utf-8
"""
defines categories based on the selection used for the trigger studies
"""

from __future__ import annotations
import order as od

from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer
from hbw.util import call_once_on_config

np = maybe_import("numpy")
ak = maybe_import("awkward")


####################################################################################################
#
# Build categories
#
####################################################################################################
# categorizer muon channel
@categorizer(uses={"Muon.pt"}, call_force=True)
def catid_trigger_mu(
    self: Categorizer,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:

    mask = (ak.sum(events.Muon.pt > 15, axis=1) >= 1)
    return events, mask


# categorizer electron channel
@categorizer(uses={"Electron.pt"}, call_force=True)
def catid_trigger_ele(
    self: Categorizer,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:

    mask = (ak.sum(events.Electron.pt > 15, axis=1) >= 1)
    return events, mask


# categorizer for orthogonal measurement (muon channel)
@categorizer(uses={"Jet.pt", "Muon.pt", "Electron.pt", "HLT.*"}, call_force=True)
def catid_trigger_orth_mu(
    self: Categorizer,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:

    mask = (
        (ak.sum(events.Muon.pt > 15, axis=1) >= 1) &
        (ak.sum(events.Electron.pt > 15, axis=1) >= 1) &
        (ak.sum(events.Jet.pt > 25, axis=1) >= 3) &
        (events.HLT[self.config_inst.x.ref_trigger["mu"]])
    )
    return events, mask


# categorizer for orthogonal measurement (electron channel)
@categorizer(uses={"Jet.pt", "Muon.pt", "Electron.pt", "HLT.*"}, call_force=True)
def catid_trigger_orth_ele(
    self: Categorizer,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:

    mask = (
        (ak.sum(events.Muon.pt > 15, axis=1) >= 1) &
        (ak.sum(events.Electron.pt > 15, axis=1) >= 1) &
        (ak.sum(events.Jet.pt > 25, axis=1) >= 3) &
        (events.HLT[self.config_inst.x.ref_trigger["e"]])
    )
    return events, mask


####################################################################################################
#
# add categories to the config
#
####################################################################################################
@call_once_on_config()
def add_trigger_categories(config: od.Config) -> None:
    """
    Adds trigger categories to the config
    """
    # mc truth categories
    cat_trig_mu = config.add_category(  # noqa
        name="trig_mu",
        id=1000,
        selection="catid_trigger_mu",
        label="Muon\n(MC truth)",
    )
    cat_trig_ele = config.add_category(  # noqa
        name="trig_ele",
        id=2000,
        selection="catid_trigger_ele",
        label="Electron\n(MC truth)",
    )
    # orthogonal categories
    cat_trig_mu_orth = config.add_category(  # noqa
        name="trig_mu_orth",
        id=3000,
        selection="catid_trigger_orth_mu",
        label="Muon\northogonal\nmeasurement",
    )
    cat_trig_ele_orth = config.add_category(  # noqa
        name="trig_ele_orth",
        id=4000,
        selection="catid_trigger_orth_ele",
        label="Electron\northogonal\nmeasurement",
    )
