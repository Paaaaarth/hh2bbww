# coding: utf-8

"""
Optimization tasks
```
"""

from __future__ import annotations

# import os

import luigi
import law

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    # SelectorMixin,
    # CalibratorsMixin,
    # ProducersMixin,
    # MLModelMixin,
    MLModelTrainingMixin,
)
from columnflow.tasks.ml import MLTraining
# from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.util import DotDict
from hbw.tasks.base import HBWTask
from hbw.tasks.ml import PlotMLResultsSingleFold


logger = law.logger.get_logger(__name__)


class GetAUCScores(PlotMLResultsSingleFold):
    """
    This is quite copy-pastey, I just need to produce some AUC scores...
    """
    data_splits = ("test",)

    def run(self):
        # imports
        from hbw.ml.data_loader import MLProcessData
        from hbw.ml.plotting import (
            plot_roc_ovr,
            plot_roc_ovo,
        )

        # prepare inputs and outputs
        inputs = self.input()
        output = self.output()
        stats = {}

        # this initializes some process information (e.g. proc_inst.x.ml_id), but feels kind of hacky
        self.ml_model_inst.datasets(self.config_inst)

        # open all model files
        training_results = self.ml_model_inst.open_model(inputs["training"])
        self.ml_model_inst.trained_model = training_results["model"]
        self.ml_model_inst.best_model = training_results["best_model"]

        process_insts = [
            self.config_inst.get_process(proc)
            for proc in self.ml_model_inst.processes
        ]

        # load data
        input_files_preml = inputs["preml"]["collection"]
        input_files_mlpred = inputs["mlpred"]["test"]["collection"]
        input_files = law.util.merge_dicts(
            *[input_files_preml[key] for key in input_files_preml.keys()],
            *[input_files_mlpred[key] for key in input_files_mlpred.keys()],
            deep=True,
        )
        data = DotDict({
            # "train": MLProcessData(self.ml_model_inst, input_files, "train", self.ml_model_inst.processes, self.fold),
            # "val": MLProcessData(self.ml_model_inst, input_files, "val", self.ml_model_inst.processes, self.fold),
            "test": MLProcessData(self.ml_model_inst, input_files, "test", self.ml_model_inst.processes, self.fold),
        })

        # ROC curves
        plot_roc_ovr(
            self.ml_model_inst,
            data["test"],
            output["plots"],
            "test",
            process_insts,
            stats,
        )
        plot_roc_ovo(
            self.ml_model_inst,
            data["test"],
            output["plots"],
            "test",
            process_insts,
            stats,
        )

        # dump all stats into yaml file
        output["stats"].dump(stats, formatter="json")


class Optimizer(
    # NOTE: mixins might need fixing, needs to be tested
    HBWTask,
    MLModelTrainingMixin,
    law.LocalWorkflow,
    # RemoteWorkflow,
):
    """
    Workflow that runs optimization. Needs to be run from within the sandbox
    cf_sandbox venv_ml_plotting
    law run hbw.Optimizer --version prod2 --ml-model dl_22
    """
    resolution_task_cls = MLTraining.resolution_task_cls
    sandbox = "bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh"

    iterations = luigi.IntParameter(default=10, description="Number of iterations")
    n_parallel = luigi.IntParameter(default=4, description="Number of parallel evaluations")
    n_initial_points = luigi.IntParameter(default=10, description="Number of random sampled values \
        before starting optimizations")

    @property
    def parameter_keys(self):
        return [
            "activation", "learningrate", "layer", "dropout",
            "batchsize", "reduce_lr_factor", "reduce_lr_patience", "epochs"
        ]

    def ask_optuna(self, study):
        trial = study.ask()
        params = {}
        params["activation"] = trial.suggest_categorical("activation", ["relu", "elu"])
        params["learningrate"] = trial.suggest_float("learningrate", 1e-6, 1e-2, log=True)
        params["layer"] = int(2**trial.suggest_int("layer_exp", 5, 10))
        params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        params["batchsize"] = int(2**trial.suggest_int("batchsize_exp", 7, 14))
        params["reduce_lr_factor"] = trial.suggest_float("reduce_lr_factor", 0.1, 1.0)
        params["reduce_lr_patience"] = trial.suggest_int("reduce_lr_patience", 1, 10)
        params["epochs"] = trial.suggest_categorical("epochs", [100])
        
        param_tuple = tuple(params[k] for k in self.parameter_keys)
        return trial, param_tuple

    def create_branch_map(self):
        return list(range(self.iterations))

    def requires(self):
        # NOTE: cache requirements?
        if self.branch == 0:
            return {}
        return Optimizer.req(self, branch=self.branch - 1)

    def workflow_requires(self):
        return {}

    def output(self):
        return self.target(f"optimizer_{self.branch}.pkl")

    def run(self):
        import optuna
        import pickle

        if self.branch == 0:
            study = optuna.create_study(direction="minimize")
        else:
            with open(self.input().path, "rb") as f:
                study = pickle.load(f)

        trials = []
        parameter_tuples = []
        for _ in range(self.n_parallel):
            trial, param_tuple = self.ask_optuna(study)
            trials.append(trial)
            parameter_tuples.append(param_tuple)

        logger.info(f"Optimizing parameters {self.parameter_keys}")
        logger.info(f"yielding Objective for sets {parameter_tuples}")
        
        output = yield Objective.req(
            self,
            parameter_keys=self.parameter_keys,
            parameter_tuples=parameter_tuples,
            iteration=self.branch,
            branch=-1,
        )
        
        y_values = [f.load()["y"] for f in output["collection"].targets.values()]

        for trial, y in zip(trials, y_values):
            study.tell(trial, y)

        print(f"minimum after {self.branch + 1} iterations: {study.best_value if len(study.trials) > 0 else 'N/A'}")

        with self.output().localize("w") as tmp:
            with open(tmp.path, "wb") as f:
                pickle.dump(study, f)


class Objective(
    # NOTE: mixins might need fixing, needs to be tested
    HBWTask,
    MLModelTrainingMixin,
    law.LocalWorkflow,
    # RemoteWorkflow,
):
    """
    Objective to optimize.
    """
    resolution_task_cls = MLTraining.resolution_task_cls

    parameter_keys = law.CSVParameter()
    parameter_tuples = law.MultiCSVParameter()
    iteration = luigi.IntParameter()

    # upstream requirements
    reqs = Requirements(
        GetAUCScores=GetAUCScores,
    )

    @property
    def parameter_dicts(self):
        """
        Convert parameter sets to dictionaries.
        """
        if hasattr(self, "_parameter_dicts"):
            return self._parameter_dicts

        parameter_dicts = []
        for i, parameter_set in enumerate(self.parameter_tuples):
            parameter_dict = dict(zip(self.parameter_keys, parameter_set))
            if "layer" in parameter_dict:
                # replace singular layer parameter with list of layers
                parameter_dict["layers"] = [parameter_dict.pop("layer")] * 3
            elif "layer1" in parameter_dict:
                # replace layer1, ..., layer{N} with list of layers
                key = "layer1"
                parameter_dict["layers"] = []
                while key in parameter_dicts.keys():
                    parameter_dict["layers"].append(parameter_dict.pop(key))
                    key = f"layer{int(key[-1]) + 1}"

            parameter_dicts.append(parameter_dict)

        self._parameter_dicts = parameter_dicts
        return self._parameter_dicts

    def create_branch_map(self):
        return {i: parameter_dict for i, parameter_dict in enumerate(self.parameter_dicts)}

    def requires(self):
        reqs = {}
        if self.branch == -1:
            return reqs
        reqs["GetAUCScores"] = self.reqs.GetAUCScores.req(
            self,
            fold=0,
            ml_model_settings=self.branch_data,
        )
        return reqs

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["GetAUCScores"] = [self.reqs.GetAUCScores.req(
            self,
            fold=0,
            ml_model_settings=parameter_dict,
            workflow="htcondor",
        ) for parameter_dict in self.parameter_dicts]
        return reqs

    def output(self):
        return self.target(f"objective_{self.iteration:02d}_{self.branch:02d}.json")

    def run(self):
        logger.info("Running Objective Task")

        # load stats
        stats = self.input()["GetAUCScores"]["stats"].load(formatter="json")

        # calculate objective value
        auc_sum = sum(stats.values()) / len(stats.values())
        objective = -auc_sum  # Negative because Optuna minimizes by default

        results = {"x": self.branch_data, "y": objective}

        # store results
        self.output().dump(results)


class DummyObjective(
    # NOTE: mixins might need fixing, needs to be tested
    HBWTask,
    MLModelTrainingMixin,
    law.LocalWorkflow,
    # RemoteWorkflow,
):
    """
    Very simple objective to minimize
    """

    parameter_keys = law.CSVParameter()
    parameter_tuples = law.MultiCSVParameter()
    iteration = luigi.IntParameter()

    @property
    def parameter_dicts(self):
        if hasattr(self, "_parameter_dicts"):
            return self._parameter_dicts

        parameter_dicts = []
        for i, parameter_tuple in enumerate(self.parameter_tuples):
            parameter_dict = dict(zip(self.parameter_keys, parameter_tuple))
            # replace singular layer parameter with list of layers
            if "layer" in parameter_dict:
                parameter_dict["layers"] = [parameter_dict.pop("layer")] * 3

            parameter_dicts.append(parameter_dict)

        self._parameter_dicts = parameter_dicts
        return self._parameter_dicts

    def create_branch_map(self):
        return {i: parameter_dict for i, parameter_dict in enumerate(self.parameter_dicts)}

    def output(self):
        return self.target(f"x_{self.iteration}_{self.branch}.json")

    def run(self):
        logger.info("Running Objective Task")

        # load some x value
        x = int(self.branch_data["layers"][0])

        # calculate some objective value (will be minimal at 512)
        y = (512 - x) ** 2
        print(x, y)
        objective = y

        # store results
        self.output().dump({"x": self.branch_data, "y": objective})
