# coding: utf-8

"""
hbw(dl) inference model.
"""

import law
from columnflow.util import DotDict
import hbw.inference.constants as const  # noqa
from hbw.inference.base import HBWInferenceModelBase


logger = law.logger.get_logger(__name__)


# patch, allowing user to fall back to old versions
use_old_version = law.config.get_expanded("analysis", "use_old_version", False)

#
# Defaults for all the Inference Model parameters
#

# used to set default requirements for cf.CreateDatacards based on the config
ml_model_name = ["multiclassv3", "ggfv3", "vbfv3"]
if use_old_version:
    ml_model_name = ["multiclassv1", "ggfv1", "vbfv1"]

# All categories to be included in the final datacard
config_categories = DotDict({
    "default": [
        "sr_2b_exact_ml_sig_all",
        "sr_2b_exact_ml_dy",
        "sr_2b_exact_ml_tt",
        "sr_2b_exact_ml_st",
        "sr_2b_exact_ml_h",
        "sr_3b_exact_ml_sig_all",
        "sr_3b_exact_ml_dy",
        "sr_3b_exact_ml_tt",
        "sr_3b_exact_ml_st",
        "sr_3b_exact_ml_h",
        "sr_4b_ml_sig_all",
        "sr_4b_ml_dy",
        "sr_4b_ml_tt",
        "sr_4b_ml_st",
        "sr_4b_ml_h",
    ],
})
# config_categories.default_boosted = (
#     config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background_resolved
# )


systematics = DotDict({
    "lumi": [
        # "lumi_13TeV_2016",
        # "lumi_13TeV_2017",
        # "lumi_13TeV_1718",
        # "lumi_13TeV_correlated",
        "lumi_13p6TeV_2022",
        "lumi_13p6TeV_2023",
    ],
    "QCDscale": [
        "QCDscale_ttbar",
        "QCDscale_V",
        "QCDscale_VV",
        "QCDscale_VVV",
        "QCDscale_ggH",
        "QCDscale_qqH",
        "QCDscale_VH",
        "QCDscale_ttH",
        # "QCDscale_bbH",
        # "QCDscale_hh_ggf",  # should be included in inference model (THU_HH)
        "QCDscale_hh_vbf",
        # "QCDscale_VHH",
        # "QCDscale_ttHH",
    ],
    "pdf": [
        "pdf_gg",
        "pdf_qqbar",
        "pdf_qg",
        "pdf_Higgs_gg",
        "pdf_Higgs_qqbar",
        # "pdf_Higgs_qg",  # none so far
        "pdf_Higgs_ttH",
        # "pdf_Higgs_bbH",  # removed
        "pdf_Higgs_hh_ggf",
        "pdf_Higgs_hh_vbf",
        # "pdf_VHH",
        # "pdf_ttHH",
    ],
    "BR": [
        "BR_hbb",
        "BR_hww",
        "BR_hzz",
        "BR_htt",
        "BR_hgg",
    ],
    "rate_unconstrained": [
        "rate_ttbar",
        "rate_dy",
    ],
    "rate_unconstrained1": [
        "rate_ttbar",
        "rate_dy_lf",
        "rate_dy_hf",
    ],
    "rate_unconstrained2": [
        "rate_ttbar"
        "rate_st",
        "rate_dy_lf",
        "rate_dy_hf",
    ],
    "rate_unconstrained3": [
        "rate_ttbar",
        "rate_ttbar_boosted",
        "rate_dy_lf",
        "rate_dy_hf",
    ],
    "rate_unconstrained_bjet_uncorr": [
        "rate_ttbar_{bjet_cat}",
        "rate_dy_{bjet_cat}",
    ],
    "hbb_efficiency": [
        "eff_hbb_signal_ggf",
        "eff_hbb_signal_vbf",
        "eff_hbb_bkg_ggf",
        "eff_hbb_bkg_vbf",
    ],
    "murf_envelope": [
        # "murf_envelope_hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "murf_envelope_ttbar",
        "murf_envelope_st",
        "murf_envelope_dy",
        # "murf_envelope_w",
        "murf_envelope_ttV",  # TODO: ttW has no murf/pdf weights
        "murf_envelope_VV",
        "murf_envelope_H",
        "murf_envelope_hh_ggf_hbb_hww",
        "murf_envelope_hh_ggf_hbb_hzz",
        "murf_envelope_hh_ggf_hbb_htt",
        # "murf_envelope_hh_vbf_hbb_hww",
        # "murf_envelope_hh_vbf_hbb_hzz",
        # "murf_envelope_hh_vbf_hbb_htt",
    ],
    "pdf_shape": [
        "pdf_shape_ttbar",
        "pdf_shape_st",
        "pdf_shape_dy",
        # "pdf_shape_w",
        "pdf_shape_ttV",  # TODO: ttW has no murf/pdf weights
        "pdf_shape_VV",
        "pdf_shape_H",
        "pdf_shape_hh_ggf_hbb_hww",
        "pdf_shape_hh_ggf_hbb_hzz",
        "pdf_shape_hh_ggf_hbb_htt",
        # "pdf_shape_hh_vbf_hbb_hww",
        # "pdf_shape_hh_vbf_hbb_hzz",
        # "pdf_shape_hh_vbf_hbb_htt",
    ],
    "btag": [
        "btag_hf",
        "btag_lf",
        "btag_hfstats1_{campaign}",
        "btag_hfstats2_{campaign}",
        "btag_lfstats1_{campaign}",
        "btag_lfstats2_{campaign}",
        "btag_cferr1",
        "btag_cferr2",
    ],
    "btag_year_uncorr": [
        "btag_hf_{year}",
        "btag_lf_{year}",
        "btag_hfstats1_{campaign}",
        "btag_hfstats2_{campaign}",
        "btag_lfstats1_{campaign}",
        "btag_lfstats2_{campaign}",
        "btag_cferr1_{year}",
        "btag_cferr2_{year}",
    ],
    "btag_bjet_uncorr": [
        "btag_hf_{bjet_cat}",
        "btag_lf_{bjet_cat}",
        "btag_hfstats1_{campaign}_{bjet_cat}",
        "btag_hfstats2_{campaign}_{bjet_cat}",
        "btag_lfstats1_{campaign}_{bjet_cat}",
        "btag_lfstats2_{campaign}_{bjet_cat}",
        "btag_cferr1_{bjet_cat}",
        "btag_cferr2_{bjet_cat}",
    ],
    "btag_cpn_uncorr": [
        "btag_hf_{campaign}",
        "btag_lf_{campaign}",
        "btag_hfstats1_{campaign}",
        "btag_hfstats2_{campaign}",
        "btag_lfstats1_{campaign}",
        "btag_lfstats2_{campaign}",
        "btag_cferr1_{campaign}",
        "btag_cferr2_{campaign}",
    ],
    "experiment": [
        "mu_id_sf",
        "mu_iso_sf",
        "e_sf",
        "e_reco_sf",
        "trigger_sf",
        "minbias_xs",
        "dy_correction",
    ],
    "experiment_cpn_uncorr": [
        "mu_id_sf_{campaign}",
        "mu_iso_sf_{campaign}",
        "e_sf_{campaign}",
        "e_reco_sf_{campaign}",
        "trigger_sf_{campaign}",
        "minbias_xs",  # do not decorrelate PU between campaigns
        "dy_correction",
    ],
    "other": [
        "isr",
        "fsr_ttbar",
        "fsr_st",
        "fsr_V",
        # "fsr_dy",
        # "fsr_w",
        "fsr_VV",
        "fsr_ttV",
        "fsr_H",  # NOTE: skip h_ggf and h_vbf because PSWeights missing in H->tautau
        "top_pt",
    ],
    "jerc_only": [
        "jer",
        "jec_Total",
    ],
    "jerc_only_bjet_uncorr": [
        "jer_{bjet_cat}",
        "jec_Total_{bjet_cat}",
    ],
    "jerc_only_cpn_uncorr": [
        "jer_{campaign}",
        "jec_Total_{campaign}",
    ],
    "jerc_only_year_uncorr": [
        "jer_{year}",
        "jec_Total_{year}",
    ],
})
systematics["rate_default"] = [
    *systematics.lumi,
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.BR,
    *systematics.hbb_efficiency,
    *systematics.rate_unconstrained3,
]
systematics["rate"] = [
    *systematics.lumi,
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.rate_unconstrained,
]
systematics["rate1"] = [
    *systematics.lumi,
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.rate_unconstrained1,
]
systematics["rate2"] = [
    *systematics.lumi,
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.rate_unconstrained2,
]
systematics["shape_only"] = [
    *systematics.murf_envelope,
    *systematics.pdf_shape,
    *systematics.btag,
    *systematics.experiment,
    *systematics.other,
]
systematics["shape_only_cpn_uncorr"] = [
    *systematics.murf_envelope,
    *systematics.pdf_shape,
    *systematics.btag_cpn_uncorr,
    *systematics.experiment_cpn_uncorr,
    *systematics.other,
]
systematics["shape"] = [
    *systematics.rate,
    *systematics.shape_only,
]
# default set of all systematics
systematics["default"] = [
    *systematics.rate_default,
    *systematics.shape_only_cpn_uncorr,
    *systematics.jerc_only_cpn_uncorr,
]
systematics["default_year_uncorr"] = [
    *systematics.rate_default,
    *systematics.murf_envelope,
    *systematics.pdf_shape,
    *systematics.btag_year_uncorr,
    *systematics.experiment,
    *systematics.other,
    *systematics.jerc_only_year_uncorr,
]
systematics["default_cpn_corr"] = [
    *systematics.rate_default,
    *systematics.shape_only,
    *systematics.jerc_only,
]

# different variations of systematic combinations (testing)
systematics["jerc"] = [
    *systematics.rate,
    *systematics.shape_only,
    *systematics.jerc_only,
]
systematics["jerc1"] = [
    *systematics.rate1,
    *systematics.shape_only,
    *systematics.jerc_only,
]
systematics["jerc2"] = [
    *systematics.rate1,
    *systematics.shape_only,
    *systematics.jerc_only_cpn_uncorr,
]
systematics["jerc3"] = [  # default fullsyst result
    *systematics.rate1,
    *systematics.shape_only_cpn_uncorr,
    *systematics.jerc_only_cpn_uncorr,
]
systematics["jerc4"] = [
    *systematics.rate2,
    *systematics.shape_only_cpn_uncorr,
    *systematics.jerc_only_cpn_uncorr,
]
systematics["rate_bjet_uncorr"] = [
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.rate_unconstrained_bjet_uncorr,
]
systematics["shape_bjet_uncorr"] = [
    *systematics.rate_bjet_uncorr,
    *systematics.murf_envelope,
    *systematics.pdf_shape,
    *systematics.btag_bjet_uncorr,
    *systematics.experiment,
    *systematics.other,
]
# All systematics with btag and rate uncertainites decorrelated between bjet categories
systematics["jerc_bjet_uncorr"] = [
    *systematics.rate_bjet_uncorr,
    *systematics.shape_bjet_uncorr,
    *systematics.jerc_only,
]
systematics["jerc_bjet_uncorr1"] = [
    *systematics.rate_bjet_uncorr,
    *systematics.shape_bjet_uncorr,
    *systematics.jerc_only_bjet_uncorr,
]

hhprocs_ggf = lambda hhdecay: [
    f"hh_ggf_{hhdecay}_kl0_kt1",
    f"hh_ggf_{hhdecay}_kl1_kt1",
    f"hh_ggf_{hhdecay}_kl2p45_kt1",
    f"hh_ggf_{hhdecay}_kl5_kt1",
]
hhprocs_vbf = lambda hhdecay: [
    f"hh_vbf_{hhdecay}_kv1p74_k2v1p37_kl14p4",
    f"hh_vbf_{hhdecay}_kvm0p758_k2v1p44_klm19p3",
    f"hh_vbf_{hhdecay}_kvm0p012_k2v0p03_kl10p2",
    f"hh_vbf_{hhdecay}_kv2p12_k2v3p87_klm5p96",
    f"hh_vbf_{hhdecay}_kv1_k2v1_kl1",
    f"hh_vbf_{hhdecay}_kv1_k2v0_kl1",  # missing bbtt sample
    f"hh_vbf_{hhdecay}_kvm0p962_k2v0p959_klm1p43",
    f"hh_vbf_{hhdecay}_kvm1p21_k2v1p94_klm0p94",
    f"hh_vbf_{hhdecay}_kvm1p6_k2v2p72_klm1p36",
    f"hh_vbf_{hhdecay}_kvm1p83_k2v3p57_klm3p39",  # missing bbtt sample
]
hhprocs = lambda hhdecay: [*hhprocs_ggf(hhdecay), *hhprocs_vbf(hhdecay)]

backgrounds = [
    "st_tchannel",
    "st_twchannel",
    "st_schannel",
    "tt",
    "ttw",
    "ttz",
    "dy_hf",
    "dy_lf",
    "w_lnu",
    "vv",
    "vvv",
    "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    "thq", "thw", "ttvh",
    "tttt",
    "ttvv",
    # TODO: add bbh
    # "qcd",  # probably not needed
]
backgrounds_skip_dy = [
    "st_tchannel",
    "st_twchannel",
    "st_schannel",
    "tt",
    "ttw",
    "ttz",
    "w_lnu",
    "vv",
    "vvv",
    "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    "thq", "thw", "ttvh",
    "tttt",
    "ttvv",
]

processes_dict = {
    "test": ["tt", *hhprocs("hbb_hww2l2nu")],
    "hww": [*backgrounds, *hhprocs("hbb_hww")],
    "hww2l2nu": [*backgrounds, *hhprocs("hbb_hww2l2nu")],
    "hwwzztt": [*backgrounds, *hhprocs("hbb_hww"), *hhprocs("hbb_hzz"), *hhprocs("hbb_htt")],
    "hwwzztt_skip_dy": [*backgrounds_skip_dy, *hhprocs("hbb_hww"), *hhprocs("hbb_hzz"), *hhprocs("hbb_htt")],
    "hwwzztt_ggf": [*backgrounds, *hhprocs_ggf("hbb_hww"), *hhprocs_ggf("hbb_hzz"), *hhprocs_ggf("hbb_htt")],
    "hhh": [
        # Add signal processes here
        "hhh_4b2w2l2nu_c30_d40",
        "hhh_4b2w2l2nu_c30_d499", "hhh_4b2w2l2nu_c30_d4m1",
        "hhh_4b2w2l2nu_c319_d419", "hhh_4b2w2l2nu_c31_d40", "hhh_4b2w2l2nu_c31_d42",
        "hhh_4b2w2l2nu_c32_d4m1", "hhh_4b2w2l2nu_c34_d49", "hhh_4b2w2l2nu_c3m1_d40",
        "hhh_4b2w2l2nu_c3m1_d4m1", "hhh_4b2w2l2nu_c3m1p5_d4m0p5",
        "st_twchannel",
        "tt",
        "ttz",
        "dy",
        "vv",
        "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
        ]

}

from hbw.ml.derived.dl import input_features
mli_inputs = input_features.v2


def config_variable_binary_ggf_and_vbf(self, config_cat_inst):
    """
    Function to set the config variable for the binary model.
    """
    if config_cat_inst.name == "sr__boosted":
        return "logit_mlscore.sig_vbf_binary"
    if "sig_ggf" in config_cat_inst.name:
        return "logit_mlscore.sig_ggf_binary"
    elif "sig_vbf" in config_cat_inst.name:
        return "logit_mlscore.sig_vbf_binary"
    elif "ggf" in config_cat_inst.name or "vbf" in config_cat_inst.name:
        return f"logit_mlscore.{config_cat_inst.x.root_cats.get('dnn').replace('ml_', '')}"
    elif config_cat_inst.x.root_cats.get("dnn"):
        # since we merge into 1 bin anyways, we can use either score
        return "logit_mlscore.sig_ggf_binary"
    else:
        # raise ValueError(f"Category {config_cat_inst.name} is not a DNN category.")
        logger.warning(
            f"Category {config_cat_inst.name} is not a DNN category, using binary classifier score.",
        )
        return "logit_mlscore.sig_ggf_binary"


default_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes_dict["hhh"],
    "config_categories": config_categories.default,
    "systematics": systematics.rate,
    "config_variable": config_variable_binary_ggf_and_vbf,
    "mc_stats": True,
    "skip_data": True,
}

dl = HBWInferenceModelBase.derive("dl", cls_dict=default_cls_dict)

#
# currently "final" inference models
#
dl_22post_multi_binary = dl.derive("dl_22post_multi_binary", cls_dict={
    "ml_model_name": ["dl_22post_multi", "dl_22post_binary"],
    "config_variable": lambda self, config_cat_inst: "logit_mlscore.sig_binary",
    "systematics": systematics.default,
})
dl_22pre_multi_binary = dl.derive("dl_22pre_multi_binary", cls_dict={
    "ml_model_name": ["dl_22pre_multi", "dl_22pre_binary"],
    "config_variable": lambda self, config_cat_inst: "logit_mlscore.sig_binary",
    "systematics": systematics.default,
})