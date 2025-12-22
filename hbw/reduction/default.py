# coding: utf-8

"""
Exemplary reduction methods that can run on-top of columnflow's default reduction.
"""
import law

from columnflow.reduction import Reducer, reducer
from columnflow.reduction.default import cf_default
from columnflow.util import maybe_import

from hbw.util import IF_TOP, IF_VJETS, IF_DY
from columnflow.production.cms.top_pt_weight import gen_parton_top
from hbw.production.gen_v import gen_v_boson
from columnflow.production.cms.dy import gen_dilepton, recoil_corrected_met

ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@reducer(
    uses={
        cf_default,
        IF_TOP(gen_parton_top),
        IF_VJETS(gen_v_boson),
        IF_DY(gen_dilepton, recoil_corrected_met),
    },
    produces={
        cf_default,
        IF_TOP(gen_parton_top),
        IF_VJETS(gen_v_boson),
        IF_DY(gen_dilepton, recoil_corrected_met),
    },
)
def default(self: Reducer, events: ak.Array, selection: ak.Array, **kwargs) -> ak.Array:
    # run cf's default reduction which handles event selection and collection creation
    events = self[cf_default](events, selection, **kwargs)

    # compute and store additional columns after the default reduction
    # (so only on a subset of the events and objects which might be computationally lighter)

    # compute gen information that will later be needed for top pt reweighting
    if self.has_dep(gen_parton_top):
        events = self[gen_parton_top](events, **kwargs)

    # compute gen information that will later be needed for vector boson pt reweighting
    if self.has_dep(gen_v_boson):
        events = self[gen_v_boson](events, **kwargs)
    if self.has_dep(gen_dilepton):
        events = self[gen_dilepton](events, **kwargs)
    if self.has_dep(recoil_corrected_met):
        events = self[recoil_corrected_met](events, **kwargs)

    return events

@default.init
def default_init(self: Reducer) -> None:
    """
    Initialize the default reducer.
    """
    # Add shift dependencies
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("jec", "jer"))
    }


triggersf = default.derive("triggersf")


@triggersf.init
def triggersf_init(self: Reducer) -> None:
    cfg = self.config_inst

    # prevent multiple initializations
    flag = f"reducer_init_done_{self.cls_name}"
    if cfg.has_tag(flag):
        return
    cfg.add_tag(flag)

    # add config entries needed already during the reduction
    # TODO: At some point this should probably time dependent as well
    cfg.x.dl_orthogonal_trigger = "PFMETNoMu120_PFMHTNoMu120_IDTight"
    cfg.x.dl_orthogonal_trigger2 = "PFMET120_PFMHT120_IDTight"
    cfg.x.hlt_L1_seeds = {
        "PFMETNoMu120_PFMHTNoMu120_IDTight": [
            "ETMHF90",
            "ETMHF100",
            "ETMHF110",
            "ETMHF120",
            "ETMHF130",
            "ETMHF140",
            "ETMHF150",
            "ETM150",
            "ETMHF90_SingleJet60er2p5_dPhi_Min2p1",
            "ETMHF90_SingleJet60er2p5_dPhi_Min2p6",
        ],
        "PFMET120_PFMHT120_IDTight": [
            "ETMHF90",
            "ETMHF100",
            "ETMHF110",
            "ETMHF120",
            "ETMHF130",
            "ETMHF140",
            "ETMHF150",
            "ETM150",
            "ETMHF90_SingleJet60er2p5_dPhi_Min2p1",
            "ETMHF90_SingleJet60er2p5_dPhi_Min2p6",
        ],
    }

    # set default hist producer
    self.config_inst.x.default_hist_producer = "default"


@triggersf.post_init
def triggersf_post_init(self: Reducer, task: law.Task, **kwargs) -> None:
    if task.selector_steps:
        raise Exception("Selector steps are not supported in triggersf reducer")

    # the updates to selector_steps and used columns are only necessary if the task invokes the reducer
    if not task.invokes_reducer:
        return

    task.selector_steps = ("all_but_trigger",)

    triggersf_required_columns = {
        f"HLT.{self.config_inst.x.dl_orthogonal_trigger}",
        *{
            f"L1.{seed}"
            for seed in self.config_inst.x.hlt_L1_seeds[self.config_inst.x.dl_orthogonal_trigger]
        },
        f"HLT.{self.config_inst.x.dl_orthogonal_trigger2}",
        *{
            f"L1.{seed}"
            for seed in self.config_inst.x.hlt_L1_seeds[self.config_inst.x.dl_orthogonal_trigger2]
        },
    }
    self.uses.update(triggersf_required_columns)
    self.produces.update(triggersf_required_columns)


triggersffix = triggersf.derive("triggersffix")
