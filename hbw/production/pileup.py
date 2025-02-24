# coding: utf-8

"""
Column production methods related to pileup weights.
"""

import functools
import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={"Pileup.nTrueInt"},
    produces={"pu_weight", "pu_weight_minbias_xs_up", "pu_weight_minbias_xs_down"},
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_pileup_file=(lambda self, external_files: external_files.pu_sf),
)
def pu_weight_from_correctionlib(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Based on the number of primary vertices, assigns each event pileup weights using correctionlib.
    """
    # compute the indices for looking up weights
    indices = events.Pileup.nTrueInt.to_numpy().astype("int32") - 1

    # map the variable names from the corrector to our columns
    variable_map = {
        "NumTrueInteractions": indices,
    }

    for column_name, syst in (
        ("pu_weight", "nominal"),
        ("pu_weight_minbias_xs_up", "up"),
        ("pu_weight_minbias_xs_down", "down"),
    ):
        # get the inputs for this type of variation
        variable_map["weights"] = syst
        inputs = [variable_map[inp.name] for inp in self.pileup_corrector.inputs]

        # evaluate and store the produced column
        pu_weight = self.pileup_corrector.evaluate(*inputs)
        events = set_ak_column(events, column_name, pu_weight, value_type=np.float32)

    return events


@pu_weight_from_correctionlib.requires
def pu_weight_from_correctionlib_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    """
    Adds the requirements needed the underlying task to derive the pileup weights into *reqs*.
    """
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@pu_weight_from_correctionlib.setup
def pu_weight_from_correctionlib_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    """
    Loads the pileup weights added through the requirements and saves them in the
    py:attr:`pu_weights` attribute for simpler access in the actual callable.
    """
    bundle = reqs["external_files"]

    # create the corrector
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    correction_set = correctionlib.CorrectionSet.from_string(
        self.get_pileup_file(bundle.files).load(formatter="gzip").decode("utf-8"),
    )

    # check
    if len(correction_set.keys()) != 1:
        raise Exception("Expected exactly one type of pileup correction")

    corrector_name = list(correction_set.keys())[0]
    self.pileup_corrector = correction_set[corrector_name]

    # check versions
    if self.pileup_corrector.version not in (0,):
        raise Exception(f"unsuppprted pileup corrector version {self.pileup_corrector.version}")
