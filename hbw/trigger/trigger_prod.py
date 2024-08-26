# coding: utf-8

"""
Column production methods related trigger studies
"""


from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.categories import category_ids

from hbw.trigger.categories import add_trigger_categories
from hbw.weight.default import default_weight_producer
from hbw.production.weights import event_weights

np = maybe_import("numpy")
ak = maybe_import("awkward")


# produce trigger column for triggers included in the NanoAOD
@producer(
    produces={"trig_bits_mu", "trig_bits_e"},
    channel=["mu", "e"],
    version=1,
)
def trigger_prod(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces column filled for each event with the triggers triggering the event.
    This column can then be used to fill a Histogram where each bin corresponds to a certain trigger.
    """

    # TODO: check if trigger were fired by unprescaled L1 seed

    for channel in self.channel:

        trig_bits = ak.Array([["allEvents"]] * len(events))

        for trigger in self.config_inst.x.trigger[channel]:

            trig_passed = ak.where(events.HLT[trigger], [[trigger]], [[]])
            trig_bits = ak.concatenate([trig_bits, trig_passed], axis=1)

        events = set_ak_column(events, f"trig_bits_{channel}", trig_bits)

    return events


# initialize the trigger producer, triggers can be set in the trigger config
@trigger_prod.init
def trigger_prod_init(self: Producer) -> None:

    for channel in self.channel:
        for trigger in self.config_inst.x.trigger[channel]:
            self.uses.add(f"HLT.{trigger}")
        self.uses.add(f"HLT.{self.config_inst.x.ref_trigger[channel]}")


# producers for single channels
mu_trigger_prod = trigger_prod.derive("mu_trigger_prod", cls_dict={"channel": ["mu"]})
ele_trigger_prod = trigger_prod.derive("ele_trigger_prod", cls_dict={"channel": ["e"]})


# add categories used for trigger studies
@producer(
    uses=category_ids,
    produces=category_ids,
    version=1,
)
def trig_cats(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Reproduces the category ids to include the trigger categories
    """

    events = self[category_ids](events, **kwargs)

    return events


@trig_cats.init
def trig_cats_init(self: Producer) -> None:

    add_trigger_categories(self.config_inst)


# always plot trig_weights with '--weight-producer no_weights', otherwise the weights themselves are weighted
@producer(
    uses={
        default_weight_producer,
        event_weights,
    },
    produces={
        "trig_weights",
    },
    version=1,
)
def trig_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces a weight column to check the event weights
    """

    events = self[event_weights](events, **kwargs)
    events, weights = self[default_weight_producer](events, **kwargs)
    events = set_ak_column(events, "trig_weights", weights)

    return events
