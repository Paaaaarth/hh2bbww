# coding: utf-8

"""
ML models using the MLClassifierBase and Mixins
"""

from __future__ import annotations

from columnflow.types import Union

import law

from columnflow.util import maybe_import

from hbw.ml.base import MLClassifierBase
from hbw.ml.mixins import DenseModelMixin, ModelFitMixin


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


class DenseClassifierDL(DenseModelMixin, ModelFitMixin, MLClassifierBase):

    combine_processes = ()

    _default__processes: tuple = (
        "hhh_4b2w2l2nu_c30_d40",
        "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
        "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
        "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
        "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
        "tt",
        "st",
        "dy",
        "h",
        # "hh_ggf_kl1_kt1",
        # "vv",
    )
    train_nodes: dict = {
        "sig_all": {
            "ml_id": 0,
            "label": r"HHH_{ALL}",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": (
                "hhh_4b2w2l2nu_c30_d40",
                "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
                "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
                "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
                "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
            ),
        },
        "tt": {"ml_id": 1},
        "st": {"ml_id": 2},
        "dy": {"ml_id": 3},
        "h": {"ml_id": 4},
        # "vv": {"ml_id": 5},
        # "hh_ggf_kl1_kt1": {"ml_id": 5},
    }

    _default__class_factors: dict = {
        "sig_all": 1,
        "tt": 1,
        "st": 1,
        "dy": 1,
        "h": 1,
        # "hh_ggf_kl1_kt1": 1,
        # "vv": 1,
    }

    _default__sub_process_class_factors = {
        "hhh_4b2w2l2nu_c30_d40": 1,
        "hhh_4b2w2l2nu_c3m1_d4m1": 1,
        "hhh_4b2w2l2nu_c30_d499": 1,
        "hhh_4b2w2l2nu_c30_d4m1": 1,
        "hhh_4b2w2l2nu_c319_d419": 1,
        "hhh_4b2w2l2nu_c31_d40": 1,
        "hhh_4b2w2l2nu_c31_d42": 1,
        "hhh_4b2w2l2nu_c32_d4m1": 1,
        "hhh_4b2w2l2nu_c34_d49": 1,
        "hhh_4b2w2l2nu_c3m1_d40": 1,
        "hhh_4b2w2l2nu_c3m1p5_d4m0p5": 1,
    }

    input_features = [
        # event features
        "mli_ht", "mli_n_jet", "mli_n_btag",
        "mli_b_score_sum",
        "mli_mindr_jj", "mli_maxdr_jj",
        # bb system
        "mli_deta_bb", "mli_dphi_bb", "mli_bb_pt", "mli_mindr_lb",
        "mli_mbb",
        # ll system
        "mli_mll", "mli_dphi_ll", "mli_deta_ll", "mli_ll_pt",
        # HH system
        "mli_min_dr_llbb",
        "mli_dphi_bb_nu", "mli_dphi_bb_llMET", "mli_mllMET",
        "mli_mbbllMET", "mli_dr_bb_llMET",
        # VBF features
        "mli_vbf_deta", "mli_vbf_mass", "mli_vbf_tag",
        # low-level features
        "mli_met_pt",
        # 4b system
        "mli_mbb_sum", "mli_mbb_sum_2", "mli_mbb_dr_sum",
        "mli_mbb_dr_sum_2", "mli_mbb_remaining", "mli_mbb_dr_max_sum",
        "mli_mbb_dr_all_sum", "mli_hh_dr","mli_dr_h_ll",
        # Test
        "mli_lb_indv_pt", "mli_lb_pt", "mli_lb_indv_mass", "mli_lb_mass",
        "mli_lb_pt_2l", "mli_lb_mass_2l", "mli_lb_indv_pt_2l", "mli_lb_indv_mass_2l",
        "mli_lb_top", "mli_lb_top_indv", "mli_lb_top_2l", "mli_lb_top_indv_2l",
        "mli_lb_top_2b", "mli_lb_top_indv_2b",
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["b1", "b2", "b3", "b4", "j1", "j2"]
        for var in ["pt", "eta", "b_score"]
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["lep", "lep2"]
        for var in ["pt", "eta"]
    ]

    store_name: str = "inputs_inclusive"

    folds: int = 5
    negative_weights: str = "ignore"

    # overwriting DenseModelMixin parameters
    _default__activation: str = "relu"
    _default__layers: tuple = (512, 512, 512)
    _default__dropout: float = 0.20
    _default__learningrate: float = 0.00050

    # overwriting ModelFitMixin parameters
    _default__callbacks: set = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True
    _default__reduce_lr_factor: float = 0.8
    _default__reduce_lr_patience: int = 3
    _default__epochs: int = 100
    _default__batchsize: int = 2 ** 12
    steps_per_epoch: Union[int, str] = "iter_smallest_process"

    # parameters to add into the `parameters` attribute to determine the 'parameters_repr' and to store in a yaml file
    bookkeep_params: set[str] = {
        # base params
        "data_loader", "input_features", "train_val_test_split",
        "processes", "sub_process_class_factors", "class_factors", "train_nodes",
        "negative_weights", "folds",
        # DenseModelMixin
        "activation", "layers", "dropout", "learningrate",
        # ModelFitMixin
        "callbacks", "reduce_lr_factor", "reduce_lr_patience",
        "epochs", "batchsize",
    }

    # parameters that can be overwritten via command line
    settings_parameters: set[str] = {
        # base params
        "processes", "class_factors", "sub_process_class_factors",
        # DenseModelMixin
        "activation", "layers", "dropout", "learningrate",
        # ModelFitMixin
        "callbacks", "reduce_lr_factor", "reduce_lr_patience",
        "epochs", "batchsize",
    }

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def cast_ml_param_values(self):
        super().cast_ml_param_values()

    def setup(self) -> None:
        super().setup()


#
# configs
#

processes = {
    "default": DenseClassifierDL._default__processes,
    "merge_hh": ["sig_all", "tt", "st", "dy", "h"]#, "hh_ggf_kl1_kt1"], # , "vv"],
}
input_features = {
    "default": DenseClassifierDL.input_features,
    "reduced": [
        # event features
        "mli_ht", "mli_n_jet", # "mli_n_btag",
        "mli_b_score_sum",
        # bb system
        "mli_dr_bb", "mli_dphi_bb", "mli_mbb", "mli_bb_pt",
        "mli_mindr_lb",
        # ll system
        "mli_mll", "mli_dr_ll", "mli_dphi_ll", "mli_ll_pt",
        "mli_min_dr_llbb",
        "mli_dphi_bb_nu", "mli_dphi_bb_llMET", "mli_mllMET",
        "mli_mbbllMET", "mli_dr_bb_llMET",
        "mli_met_pt",
        # 4b system
        "mli_mbb_sum", "mli_mbb_sum_2", "mli_mbb_dr_sum",
        "mli_mbb_dr_sum_2", "mli_mbb_remaining", "mli_mbb_dr_max_sum",
        "mli_mbb_dr_all_sum", "mli_hh_dr","mli_dr_h_ll",
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["b1", "b2", "b3", "b4", "j1", "j2"]
        for var in ["pt", "eta"] #, "b_score"]
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["lep", "lep2"]
        for var in ["pt", "eta"]
    ],
}
class_factors = {
    "default": DenseClassifierDL._default__class_factors,
    "ones": {},  # defaults to 1 (NOTE: do not try to use defaultdict! does not work with hash generation)
    "benchmark": {
        "sig_all": 1,
        "tt": 8,
        "st": 2,
        "dy": 2,
        "h": 1,
        # "hh_ggf_kl1_kt1": 1,
        # "vv": 1,
    },
}
#
# derived MLModels
#

dl_22post = DenseClassifierDL.derive("dl_22post", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
})
dl_22pre = DenseClassifierDL.derive("dl_22pre", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22pre"],
})
dl_22 = DenseClassifierDL.derive("dl_22", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post", "c22pre"],
})
# v0: using MultiDataset for validation, looping 2x
# v1: modifying MultiDataset to loop over all events in validation, but incorrect weights
# v2: use validation_weights that reweight sum of weights to the requested batchsize
# v3: final ?
dl_22post_benchmark_v3 = DenseClassifierDL.derive("dl_22post_benchmark_v3", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "class_factors": class_factors["benchmark"],
})

############# Multi classifiers ###############

dl_22post_first = dl_22post.derive("dl_22post_first", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "class_factors": class_factors["ones"],
    "input_features": input_features["reduced"] + [ "mli_lb_pt", 
                                                    "mli_lb_mass_2l", 
                                                    "mli_lb_indv_pt_2l",],
})

dl_22pre_first = dl_22pre.derive("dl_22pre_first", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22pre"],
    "class_factors": class_factors["ones"],
    "input_features": input_features["reduced"] + [ "mli_lb_pt", 
                                                    "mli_lb_mass_2l", 
                                                    "mli_lb_indv_pt_2l",],
})

dl_22post_large = dl_22post.derive("dl_22post_large", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "class_factors": class_factors["ones"],
    "input_features": input_features["reduced"],
    "layers": (1024, 1024, 1024, 1024),
})

test = dl_22post.derive("test", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"], 
    "class_factors": class_factors["ones"],
    "input_features": input_features["reduced"] + [ "mli_lb_pt", 
                                                    "mli_lb_mass_2l", 
                                                    "mli_lb_indv_pt_2l",],
    "layers": (512, 256, 128, 64),
})


dl_22post_limited = dl_22post.derive("dl_22post_limited", cls_dict={
    "training_configs": lambda self, requested_configs: ["l22post"],
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "st_tchannel_t"],
    "epochs": 6,
})

############# Binary classifiers ###############

dl_22post_binary = dl_22post.derive("dl_22post_binary", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"], 
    "input_features": input_features["reduced"] + [ "mli_lb_pt", 
                                                    "mli_lb_mass_2l", 
                                                    "mli_lb_indv_pt_2l",],
    "train_nodes": {
        "sig_binary": {
            "ml_id": 0,
            "label": "Signal",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": (
                "hhh_4b2w2l2nu_c30_d40", "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
                "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
                "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
                "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
            ),
        },
        "bkg_binary": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": (
                "tt",
                "st",
                "dy",
                "h",
            ),
        },
    },
    # relative class factors between different nodes
    "class_factors": {
        "sig_binary": 1,
        "bkg_binary": 1,
    },
    "epochs": 100,
})

dl_22post_binary_sm = dl_22post.derive("dl_22post_binary_sm", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"], 
    "input_features": input_features["reduced"] + [ "mli_lb_pt", 
                                                    "mli_lb_mass_2l", 
                                                    "mli_lb_indv_pt_2l",],
    "train_nodes": {
        "sig_binary": {
            "ml_id": 0,
            "label": "Signal",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": (
                "hhh_4b2w2l2nu_c30_d40", "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
                "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
                "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
                "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
            ),
        },
        "bkg_binary": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": (
                "tt",
                "st",
                "dy",
                "h",
            ),
        },
    },
    # relative class factors between different nodes
    "class_factors": {
        "sig_binary": 1,
        "bkg_binary": 1,
    },
        # relative process weights within one class
    "sub_process_class_factors": {
        "hhh_4b2w2l2nu_c30_d40": 2,
        "hhh_4b2w2l2nu_c30_d499": 1,
        "hhh_4b2w2l2nu_c30_d4m1": 1,
        "hhh_4b2w2l2nu_c319_d419": 1,
        "hhh_4b2w2l2nu_c31_d40": 1,
        "hhh_4b2w2l2nu_c31_d42": 1,
        "hhh_4b2w2l2nu_c32_d4m1": 1,
        "hhh_4b2w2l2nu_c34_d49": 1,
        "hhh_4b2w2l2nu_c3m1_d40": 1,
        "hhh_4b2w2l2nu_c3m1_d4m1": 1,
        "hhh_4b2w2l2nu_c3m1p5_d4m0p5": 1,
        "tt": 1,
        "st": 1,
        "dy": 1,
        "h": 1,
    },
    "epochs": 100,
})

dl_22pre_binary = dl_22pre.derive("dl_22pre_binary", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22pre"], 
    "input_features": input_features["reduced"] + [ "mli_lb_pt", 
                                                    "mli_lb_mass_2l", 
                                                    "mli_lb_indv_pt_2l",],
    "train_nodes": {
        "sig_binary": {
            "ml_id": 0,
            "label": "Signal",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": (
                "hhh_4b2w2l2nu_c30_d40", "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
                "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
                "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
                "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
            ),
        },
        "bkg_binary": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": (
                "tt",
                "st",
                "dy",
                "h",
            ),
        },
    },
    # relative class factors between different nodes
    "class_factors": {
        "sig_binary": 1,
        "bkg_binary": 1,
    },
    "epochs": 100,
})


# Combined training on c22post and c22pre

dl_22_multi = dl_22.derive("dl_22_multi", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post", "c22pre"],
    "class_factors": class_factors["ones"],
    "input_features": input_features["reduced"] + [ "mli_lb_pt", 
                                                    "mli_lb_mass_2l", 
                                                    "mli_lb_indv_pt_2l",],
})


dl_22_binary = dl_22post.derive("dl_22_binary", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post", "c22pre"], 
    "input_features": input_features["reduced"] + [ "mli_lb_pt",
                                                    "mli_lb_mass_2l", 
                                                    "mli_lb_indv_pt_2l",],
    "train_nodes": {
        "sig_binary": {
            "ml_id": 0,
            "label": "Signal",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": (
                "hhh_4b2w2l2nu_c30_d40", "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
                "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
                "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
                "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
            ),
        },
        "bkg_binary": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": (
                "tt",
                "st",
                "dy",
                "h",
            ),
        },
    },
    # relative class factors between different nodes
    "class_factors": {
        "sig_binary": 1,
        "bkg_binary": 1,
    },
    "epochs": 100,
})


dl_22post_vbf = dl_22post.derive("dl_22post_vbf", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "processes": [
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
        "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
        "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
        "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
        "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",
        "tt",
        "st",
        "dy_m4to10",
        "dy_m10to50",
        "dy_m50toinf",
        "vv",
        "ttv",
        "h",
    ],
    "train_nodes": {
        "sig_vbf_binary": {
            "ml_id": 0,
            "label": "HH VBF",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": (
                "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
                "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
                "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
                "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
                "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
                "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",
            ),
        },
        "bkg_binary_for_vbf": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": (
                "tt",
                "st",
                "dy_m4to10",
                "dy_m10to50",
                "dy_m50toinf",
                "vv",
                "ttv",
                "h",
            ),
        },
    },
    "class_factors": {
        "sig_vbf_binary": 1,
        "bkg_binary_for_vbf": 1,
    },
    # relative process weights within one class
    "sub_process_class_factors": {
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": 1,
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1": 1,
        "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43": 1,
        "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94": 1,
        "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36": 1,
        "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39": 1,
        "tt": 1,
        "st": 1,
        "dy_m4to10": 0.1,  # assign small weight due to low statistics
        "dy_m10to50": 1,
        "dy_m50toinf": 1,
        "vv": 1,
        "ttv": 1,
        "h": 1,
    },
    "epochs": 100,
})
