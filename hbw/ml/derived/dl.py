# coding: utf-8

"""
ML models using the MLClassifierBase and Mixins
"""

from __future__ import annotations

from columnflow.types import Union

import law

from columnflow.util import maybe_import, DotDict

from hbw.ml.base import MLClassifierBase
from hbw.ml.mixins import DenseModelMixin, ModelFitMixin
# from hbw.ml.mixins import ModelFitMixin, TransformerModelMixin

from hbw.config.styling import color_palette


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


# class DenseClassifierDL(ModelFitMixin, MLClassifierBase, TransformerModelMixin):
class DenseClassifierDL(DenseModelMixin, ModelFitMixin, MLClassifierBase):
    _default__processes: tuple = (
        "hhh_4b2w2l2nu_c30_d40",
        "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
        "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
        "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
        "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5", 
        "tt_custom", "ttbb_custom",
        "st",
        "dy",
        "h",
        "hh",
    )
    train_nodes: dict = {
        "sig_all": {
            "ml_id": 0,
            "label": r"HHH",
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
        "hh": {"ml_id": 1,
            "label": r"HH",},
        "h": {"ml_id": 2,
            "label": r"H",},
        "di_top": {"ml_id": 3,
            "label": r"TT",
            "sub_processes": ("tt_custom","ttbb_custom"),},
        "st": {"ml_id": 4,
            "label": r"ST",},
        "dy": {"ml_id": 5,
            "label": r"DY",},   
    }
    _default__class_factors: dict = {
        "sig_all": 1,
        "di_top": 1,
        "st": 1,
        "dy": 1,
        "h": 1,
        "hh": 1, 
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
        "mli_deta_bb", "mli_dphi_bb", "mli_mbb", "mli_bb_pt",
        "mli_mindr_lb",
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
        # test
        "mli_lb_indv_pt", "mli_lb_pt", "mli_lb_indv_mass",
        "mli_lb_mass", "mli_lb_pt_2l", "mli_lb_mass_2l", "mli_lb_indv_pt_2l", "mli_lb_indv_mass_2l",
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

    preparation_producer_name = "prepml"

    folds: int = 5
    negative_weights: str = "ignore"

    # overwriting DenseModelMixin parameters
    _default__activation: str = "relu"
    _default__layers: tuple = (1024, 1024, 1024)
    # _default__dropout: float = 0.20
    # _default__learningrate: float = 0.00050
    _default__dropout: float = 0.20683461688335403
    _default__learningrate: float = 0.0002680773983111746

    # overwriting ModelFitMixin parameters
    _default__callbacks: set = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True
    # _default__reduce_lr_factor: float = 0.8
    # _default__reduce_lr_patience: int = 3
    # _default__epochs: int = 100
    # _default__batchsize: int = 2 ** 12
    _default__reduce_lr_factor: float = 0.7595738861703812
    _default__reduce_lr_patience: int = 9
    _default__epochs: int = 100
    _default__batchsize: int = 8192
    steps_per_epoch: Union[int, str] = "iter_smallest_process"

    # parameters to add into the `parameters` attribute to determine the 'parameters_repr' and to store in a yaml file
    bookkeep_params: set[str] = {
        # base params
        "data_loader", "input_features", "train_val_test_split",
        "processes", "sub_process_class_factors", "class_factors", "train_nodes",
        "negative_weights", "folds",
        # DenseModelMixin
        "activation", "layers", "dropout", "learningrate", "l2_regularization",
        # ModelFitMixin
        "callbacks", "reduce_lr_factor", "reduce_lr_patience",
        "epochs", "batchsize",
    }

    # parameters that can be overwritten via command line
    settings_parameters: set[str] = {
        # base params
        "processes", "class_factors", "sub_process_class_factors",
        # DenseModelMixin
        "activation", "layers", "dropout", "learningrate", "l2_regularization",
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
processes = DotDict({
    "merge_hh": ["sig_all", "tt", "st", "dy", "h"],
})
input_features = DotDict({
    "default": DenseClassifierDL.input_features,
    "previous": [
        # event features
        "mli_ht", "mli_n_jet", "mli_n_btag",
        "mli_b_score_sum",
        # bb system
        "mli_dr_bb", "mli_dphi_bb", "mli_mbb", "mli_bb_pt",
        "mli_mindr_lb",
        # ll system
        "mli_mll", "mli_dr_ll", "mli_dphi_ll", "mli_ll_pt",
        "mli_min_dr_llbb",
        "mli_dphi_bb_nu", "mli_dphi_bb_llMET", "mli_mllMET",
        "mli_mbbllMET", "mli_dr_bb_llMET",
        # # VBF features
        # "mli_vbf_deta", "mli_vbf_mass", "mli_vbf_tag",
        # low-level features
        "mli_met_pt",
        # 4b system,
        "mli_mbb_sum", "mli_mbb_sum_2", "mli_mbb_dr_sum",
        "mli_mbb_dr_sum_2", "mli_mbb_remaining", "mli_mbb_dr_max_sum",
        "mli_mbb_dr_all_sum", "mli_hh_dr", "mli_dr_h_ll",
        "mli_lb_pt", "mli_lb_mass_2l", "mli_lb_indv_pt_2l",

    ] + [
        f"mli_{obj}_{var}"
        for obj in ["b1", "b2", "b3", "b4"]
        for var in ["pt", "eta", "b_score"]
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["lep", "lep2"]
        for var in ["pt", "eta"]
    ],
    "reduced": [
        "mli_mbb",
        "mli_b1_pt",
        "mli_j1_pt",
        "mli_n_jet",
        "mli_mbbllMET",
        "mli_dr_bb_llMET",
        "mli_j1_eta",
        "mli_met_pt",
        "mli_mllMET",
        "mli_mll",
        "mli_ll_pt",
        "mli_lep_pt",
        "mli_lep2_pt",
        "mli_dphi_bb_nu",  # badly modelled ---> please remove in future
        "mli_j1_b_score",
        "mli_bb_pt",
        "mli_dr_ll",
        "mli_b_score_sum",
        "mli_min_dr_llbb",
        "mli_b2_pt",
        # "mli_b3_pt",
        # "mli_b4_pt",
        # "mli_lep_eta",
        # "mli_b1_eta",
        "mli_dr_bb",
        "mli_mindr_lb",
        "mli_ht",
        # "mli_b2_eta",
        # "mli_b2_b_score",
        # "mli_n_btag",
        # "mli_lep2_eta",
        # "mli_dhpi_bb",
        # "mli_b1_b_score",
        # "mli_dhpi_bb_llMET",
        # "mli_dphi_ll",
        # 4b system,
        "mli_mbb_sum", "mli_mbb_sum_2", "mli_mbb_dr_sum",
        "mli_mbb_dr_sum_2", "mli_mbb_remaining", "mli_mbb_dr_max_sum",
        "mli_mbb_dr_all_sum", "mli_hh_dr", "mli_dr_h_ll",
        "mli_lb_pt", "mli_lb_mass_2l", "mli_lb_indv_pt_2l",

    ],
    "test": [
        "mli_b_score_sum",
        "mli_b1_pt",
        "mli_b2_pt",
        "mli_b3_eta",
        "mli_bb_pt",
        "mli_deta_ll",
        "mli_dphi_ll",
        "mli_dr_B0_B1",
        "mli_dr_B0_B2",
        "mli_dr_bb",
        "mli_dr_ll",
        "mli_ht_alljets",
        "mli_ht",
        "mli_j1_b_score",
        "mli_l_b_score_sum",
        "mli_lb_indv_mass",
        "mli_lb_mass",
        "mli_ll_pt",
        "mli_mbb_dr_sum",
        "mli_mbb_sum",
        "mli_mindr_jj",
        "mli_mindr_lj",
        "mli_mll",
        "mli_n_btag",
    ], 
    # + [
    #     "mli_fj_mass",
    #     "mli_fj_msoftdrop",
    #     "mli_fj_particleNetWithMass_HbbvsQCD",
    #     "mli_fj_particleNet_XbbVsQCD",
    # ]
})

class_factors = {
    "default": DenseClassifierDL._default__class_factors,
    "ones": {},  # defaults to 1 (NOTE: do not try to use defaultdict! does not work with hash generation)
    "benchmark": {
        "sig_all": 1,
        "tt": 1,
        "st": 1,
        "dy": 1,
        "h": 1,
        "hh": 1,
    },
}

configs = DotDict({
    "22post": lambda self, requested_configs: ["c22postv14"],
    "22": lambda self, requested_configs: ["c22prev14", "c22postv14"],
    "23": lambda self, requested_configs: ["c23prev14", "c23postv14"],
    "full": lambda self, requested_configs: ["c22prev14", "c22postv14", "c23prev14", "c23postv14"],
})

#
# derived MLModels
#

dl_22post_multi = DenseClassifierDL.derive("dl_22post_multi", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22postv14"],
    "class_factors": class_factors["ones"],
    "input_features": input_features["previous"]})

#
# category specific models for 22post
#
dl_22post_multi_2b = DenseClassifierDL.derive("dl_22post_multi_2b", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22postv14"],
    "input_features": input_features["previous"],
    # "preparation_producer_name": "prepml_2b",
    "processes": (
        "hhh_4b2w2l2nu_c30_d40",
        # "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
        # "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
        # "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
        # "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
        "hh",
        "tt_custom",
        "ttbb_custom",
        "h", "st", "dy",
    ),
    "train_nodes": {
        "sig_all": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": (
                "hhh_4b2w2l2nu_c30_d40",
                # "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
                # "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
                # "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
                # "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
            ),
        },
        "hh": {"ml_id": 1,
            "label": r"HH",},
        "h": {"ml_id": 2,
            "label": r"H",},
        "tt+st": {"ml_id": 3,
            "label": r"tt+st",
            "sub_processes": ("tt_custom","ttbb_custom", "st"),},
        "dy": {"ml_id": 4,
            "label": r"DY",},   
    },
    "class_factors": {
        "sig_all": 1,
        "tt+st": 1,
        "dy": 1,
        "h": 1,
        "hh": 1, 
    },
    })
dl_22post_multi_3b = DenseClassifierDL.derive("dl_22post_multi_3b", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22postv14"],
    "input_features": input_features["previous"],
    "preparation_producer_name": "prepml_3b",
    "processes": (
        "hhh_4b2w2l2nu_c30_d40",
        # "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
        # "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
        # "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
        # "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
        "tthh_4b",
        "tt_custom",
        "ttbb_custom",
        "tth"
    ),
    "train_nodes": {
        "sig_all": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": (
                "hhh_4b2w2l2nu_c30_d40",
                # "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
                # "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
                # "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
                # "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
            ),
        },
        "tthh_4b": {"ml_id": 1, "label": r"HH",},
        "tth": {"ml_id": 2, "label": r"H",},
        "di_top": {
            "ml_id": 3,
            "label": r"tt",
            "color": "#000000",  # black
            "class_factor_mode": "xsec",
            "sub_processes": ["tt_custom", "ttbb_custom"],
        },
    },
    "class_factors": {
        "sig_all": 11,
        "tthh_4b": 7,
        "tth": 10,
        "di_top": 12,},
    })
dl_22post_multi_4b = DenseClassifierDL.derive("dl_22post_multi_4b", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22postv14"],
    "input_features": input_features["previous"],
    "preparation_producer_name": "prepml_4b",
    "processes": (
        "hhh_4b2w2l2nu_c30_d40",
        # "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
        # "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
        # "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
        # "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
        "tthh_4b",
        "tt_custom",
        "ttbb_custom",
        "tth"
    ),
    "train_nodes": {
        "sig_all": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": (
                "hhh_4b2w2l2nu_c30_d40",
                # "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
                # "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
                # "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
                # "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
            ),
        },
        "tthh_4b": {"ml_id": 1, "label": r"HH",},
        "tth": {"ml_id": 2, "label": r"H",},
        "di_top": {
            "ml_id": 3,
            "label": r"tt",
            "color": "#000000",  # black
            "class_factor_mode": "xsec",
            "sub_processes": ["tt_custom", "ttbb_custom"],
        },
    },
    "class_factors": {
        "sig_all": 10,
        "tthh_4b": 7,
        "tth": 10,
        "di_top": 11,},
    })

test = DenseClassifierDL.derive("test", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22postv14", "c23postv14", "c22prev14", "c23prev14"],
    "class_factors": class_factors["ones"],
    "input_features": input_features["previous"],
    "preparation_producer_name": "prepml_2b",
    })

test_2b_sr = DenseClassifierDL.derive("test_2b_sr", cls_dict={
    # "training_configs": lambda self, requested_configs: ["c22postv14"],
    "training_configs": lambda self, requested_configs: ["c22postv14", "c23postv14", "c22prev14", "c23prev14"],
    "class_factors": class_factors["ones"],
    "input_features": input_features["previous"],
    "preparation_producer_name": "prepml_2b_sr",
    })

# test_2j = DenseClassifierDL.derive("test_2j", cls_dict={
#     "training_configs": lambda self, requested_configs: ["c22postv14"],
#     "input_features": input_features["previous"],
#     "preparation_producer_name": "prepml_2b",
#     "processes": (
#         "hhh_4b2w2l2nu_c30_d40",
#         # "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
#         # "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
#         # "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
#         # "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
#         "hh",
#         "tt_custom",
#         "ttbb_custom",
#         "h", "st", "dy",
#     ),
#     "train_nodes": {
#         "sig_all": {
#             "ml_id": 0,
#             "label": r"HHH",
#             "color": "#000000",  # black
#             "class_factor_mode": "equal",
#             "sub_processes": (
#                 "hhh_4b2w2l2nu_c30_d40",
#                 # "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
#                 # "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
#                 # "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
#                 # "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
#             ),
#         },
#         "hh": {"ml_id": 1,
#             "label": r"HH",},
#         "h": {"ml_id": 2,
#             "label": r"H",},
#         "tt+st": {"ml_id": 3,
#             "label": r"tt+st",
#             "sub_processes": ("tt_custom","ttbb_custom", "st"),},
#         "dy": {"ml_id": 4,
#             "label": r"DY",},   
#     },
#     "class_factors": {
#         "sig_all": 1,
#         "tt+st": 1,
#         "dy": 1,
#         "h": 1,
#         "hh": 1, 
#     },
#     })

dl_22pre_multi = DenseClassifierDL.derive("dl_22pre_multi", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22prev14"],
    "class_factors": class_factors["ones"],
    "input_features": input_features["reduced"]})

dl_23post_multi = DenseClassifierDL.derive("dl_23post_multi", cls_dict={
    "training_configs": lambda self, requested_configs: ["c23postv14"],
    "class_factors": class_factors["ones"],
    "input_features": input_features["reduced"]})
dl_23pre_multi = DenseClassifierDL.derive("dl_23pre_multi", cls_dict={
    "training_configs": lambda self, requested_configs: ["c23prev14"],
    "class_factors": class_factors["ones"],
    "input_features": input_features["reduced"]})

dl_22post_binary = DenseClassifierDL.derive("dl_22post_binary", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22postv14"], 
    "input_features": input_features["previous"],
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
                "tt_custom", "ttw", "tttt", "ttbb_custom",
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
        "tt_custom": 1,
        "ttbb_custom": 1,
        "st": 1,
        "dy": 1,
        "h": 1,
        "ttw": 1,
        "tttt": 1,
    },
    "epochs": 100,
    "processes":(
        "hhh_4b2w2l2nu_c30_d40",
        "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
        "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
        "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
        "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
        "tt_custom", "ttbb_custom", "ttw", "tttt",
        "st",
        "dy",
        "h",
    ),
})
dl_22pre_binary = DenseClassifierDL.derive("dl_22pre_binary", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22prev14"],     
    "input_features": input_features["reduced"],
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
    "processes":(
        "hhh_4b2w2l2nu_c30_d40",
        "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
        "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
        "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
        "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
        "tt",
        "st",
        "dy",
        "h",
    ),
})

dl_23post_binary = DenseClassifierDL.derive("dl_23post_binary", cls_dict={
    "training_configs": lambda self, requested_configs: ["c23postv14"], 
    "input_features": input_features["reduced"],
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
    "processes":(
        "hhh_4b2w2l2nu_c30_d40",
        "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
        "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
        "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
        "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
        "tt",
        "st",
        "dy",
        "h",
    ),
})
dl_23pre_binary = DenseClassifierDL.derive("dl_23pre_binary", cls_dict={
    "training_configs": lambda self, requested_configs: ["c23prev14"], 
    "input_features": input_features["reduced"],
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
    "processes":(
        "hhh_4b2w2l2nu_c30_d40",
        "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
        "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
        "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
        "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
        "tt",
        "st",
        "dy",
        "h",
    ),
})


dl_combo_multi = DenseClassifierDL.derive("dl_combo_multi", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22prev14","c22postv14","c23prev14", "c23postv14"],
    "class_factors": class_factors["ones"],
    "input_features": input_features["previous"]})

dl_combo_binary = DenseClassifierDL.derive("dl_combo_binary", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22prev14","c22postv14","c23prev14", "c23postv14"], 
    "input_features": input_features["previous"],
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
    "processes":(
        "hhh_4b2w2l2nu_c30_d40",
        "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
        "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
        "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
        "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
        "tt",
        "st",
        "dy",
        "h",
    ),
})


####################
### 4-class multi-classifiers
####################

full_multi_2b = DenseClassifierDL.derive("full_multi_2b", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22postv14", "c23postv14", "c22prev14", "c23prev14"],
    "input_features": input_features["previous"],
    "preparation_producer_name": "prepml_2b",
    "processes": (
        "hhh_4b2w2l2nu_c30_d40",
        "hh",
        "tt_custom",
        "ttbb_custom",
        "h", "st", "dy",
    ),
    "train_nodes": {
        "hhh_4b2w2l2nu_c30_d40": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
        },
        "hh": {"ml_id": 1,
            "label": r"HH",},
        "h": {"ml_id": 2,
            "label": r"H",},
        "tt+st": {"ml_id": 3,
            "label": r"tt+st",
            "sub_processes": ("tt_custom","ttbb_custom", "st"),},
        "dy": {"ml_id": 4,
            "label": r"DY",},   
    },
    "class_factors": {
        "hhh_4b2w2l2nu_c30_d40": 1,
        "tt+st": 1,
        "dy": 1,
        "h": 1,
        "hh": 1, 
    },
    })

full_multi_3b = DenseClassifierDL.derive("full_multi_3b", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22postv14", "c23postv14", "c22prev14", "c23prev14"],
    "input_features": input_features["previous"],
    "preparation_producer_name": "prepml_3b",
    "processes": (
        "hhh_4b2w2l2nu_c30_d40",
        "tthh_4b",
        "tt_custom",
        "ttbb_custom",
        "tth"
    ),
    "train_nodes": {
        "hhh_4b2w2l2nu_c30_d40": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
        },
        "tthh_4b": {"ml_id": 1, "label": r"HH",},
        "tth": {"ml_id": 2, "label": r"H",},
        "di_top_3b": {
            "ml_id": 3,
            "label": r"tt",
            "color": "#000000",  # black
            "class_factor_mode": "xsec",
            "sub_processes": ["tt_custom", "ttbb_custom"],
        },
    },
    "class_factors": {
        "hhh_4b2w2l2nu_c30_d40": 11,
        "tthh_3b": 7,
        "tth": 10,
        "di_top_3b": 12,},
    })

full_multi_4b = DenseClassifierDL.derive("full_multi_4b", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22postv14", "c23postv14", "c22prev14", "c23prev14"],
    "input_features": input_features["previous"],
    "preparation_producer_name": "prepml_4b",
    "processes": (
        "hhh_4b2w2l2nu_c30_d40",
        "tthh_4b",
        "tt_custom",
        "ttbb_custom",
        "tth"
    ),
    "train_nodes": {
        "hhh_4b2w2l2nu_c30_d40": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
        },
        "tthh_4b": {"ml_id": 1, "label": r"HH",},
        "tth": {"ml_id": 2, "label": r"H",},
        "di_top_4b": {
            "ml_id": 3,
            "label": r"tt",
            "color": "#000000",  # black
            "class_factor_mode": "xsec",
            "sub_processes": ["tt_custom", "ttbb_custom"],
        },
    },
    "class_factors": {
        "hhh_4b2w2l2nu_c30_d40": 10,
        "tthh_4b": 7,
        "tth": 10,
        "di_top_4b": 11,},
    })

full_binary = DenseClassifierDL.derive("full_binary", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22postv14", "c23postv14", "c22prev14", "c23prev14"], 
    "input_features": input_features["previous"],
    "preparation_producer_name": "prepml_geq3b",
    "train_nodes": {
        "sig_binary": {
            "ml_id": 0,
            "label": "Signal",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": (
                "hhh_4b2w2l2nu_c30_d40", 
                "hhh_4b2w2l2nu_c30_d4m1",
                "hhh_4b2w2l2nu_c31_d40", 
            ),
        },
        "bkg_binary": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": (
                "tt_custom", "ttw", "tttt", 
                "ttbb_custom",
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
        "tt_custom": 1,
        "ttbb_custom": 1,
        "st": 1,
        "dy": 1,
        "h": 1,
        "ttw": 1,
        "tttt": 1,
    },
    "epochs": 100,
    "processes":(
        "hhh_4b2w2l2nu_c30_d40",
        "hhh_4b2w2l2nu_c30_d4m1",
        "hhh_4b2w2l2nu_c31_d40", 
        "tt_custom", "ttbb_custom", "ttw", "tttt",
        "st",
        "dy",
        "h",
    ),
})