# coding: utf-8

""" production methods regarding ml stats """

from __future__ import annotations

import functools

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.ml import MLModel
from columnflow.columnar_util import set_ak_column
from columnflow.selection.stats import increment_stats
from hbw.categorization.categories import catid_sr

ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

logger = law.logger.get_logger(__name__)


@producer(
    uses={catid_sr, increment_stats, "process_id", "fold_indices"},
    # produces={"fold_indices"},
)
def ml_preparation(
    self: Producer,
    events: ak.Array,
    stats: dict = {},
    fold_indices: ak.Array | None = None,
    ml_model_inst: MLModel | None = None,
    **kwargs,
) -> ak.Array:
    """
    Producer that is run as part of PrepareMLEvents to collect relevant stats
    """
    if self.task.task_family == "cf.PrepareMLEvents":
        # pass category mask to not use the full phase space in training
        events, mask = self[catid_sr](events, **kwargs)
        logger.info(f"Select {ak.sum(mask)} from {len(events)} events for MLTraining")
        events = events[mask]

    weight_map = {
        "num_events": Ellipsis,  # all events
    }

    if self.task.dataset_inst.is_mc:
        weight = events["normalization_weight"]
        stats["sum_weights"] += float(ak.sum(weight, axis=0))
        weight_map["sum_weights"] = weight
        weight_map["sum_abs_weights"] = (weight, weight > 0)
        weight_map["sum_pos_weights"] = np.abs(weight)

    group_map = {
        "process": {
            "values": events.process_id,
            "mask_fn": (lambda v: events.process_id == v),
        },
        "fold": {
            "values": events.fold_indices,
            "mask_fn": (lambda v: events.fold_indices == v),
        },
    }

    group_combinations = [("process", "fold")]

    self[increment_stats](
        events,
        None,  # SelectionResult that is not required
        stats,
        weight_map=weight_map,
        group_map=group_map,
        group_combinations=group_combinations,
        **kwargs,
    )
    return events


@ml_preparation.init
def ml_preparation_init(self):
    # TODO: we access self.task.dataset_inst instead of self.dataset_inst due to an issue
    # with the preparation producer initialization
    if not getattr(self.task, "dataset_inst", None) or self.task.dataset_inst.is_data:
        return

    self.uses.add("normalization_weight")
