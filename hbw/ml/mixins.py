# # coding: utf-8

# """
# Mixin classes to build ML models
# """

# from __future__ import annotations

# import functools
# import math

# import law
# # import order as od

# from columnflow.types import Union
# from columnflow.util import maybe_import, DotDict
# from hbw.util import log_memory, call_func_safe, timeit


# np = maybe_import("numpy")
# ak = maybe_import("awkward")

# logger = law.logger.get_logger(__name__)


# def loop_dataset(data, max_count=10000):
#     for i, x in enumerate(data):
#         if i % int(max_count / 100) == 0:
#             print(i)
#         if i == max_count:
#             break


# class DenseModelMixin(object):
#     """
#     Mixin that provides an implementation for `prepare_ml_model`
#     """

#     _default__activation: str = "relu"
#     _default__layers: tuple[int] = (64, 64, 64)
#     _default__dropout: float = 0.50
#     _default__learningrate: float = 0.00050

#     # TODO: these parameters are currently not part of the MLModel repr
#     loss: str = "categorical_crossentropy"
#     focal_loss_alpha: float = 0.25
#     focal_loss_gamma: float = 2.0

#     def cast_ml_param_values(self):
#         """
#         Cast the values of the parameters to the correct types
#         """
#         super().cast_ml_param_values()
#         self.activation = str(self.activation)
#         self.layers = tuple(int(n_nodes) for n_nodes in self.layers)
#         self.dropout = float(self.dropout)
#         self.learningrate = float(self.learningrate)

#         self.loss = str(self.loss)
#         self.focal_loss_alpha = float(self.focal_loss_alpha)
#         self.focal_loss_gamma = float(self.focal_loss_gamma)

#     def prepare_ml_model(
#         self,
#         task: law.Task,
#     ):
#         import tensorflow.keras as keras
#         from keras.models import Sequential
#         from keras.layers import Dense, BatchNormalization
#         from hbw.ml.tf_util import cumulated_crossentropy
#         # from keras.losses import CategoricalFocalCrossentropy

#         n_inputs = len(set(self.input_features))
#         # n_outputs = len(self.processes)
#         n_outputs = len(self.train_nodes.keys())

#         # define the DNN model
#         model = Sequential()

#         # BatchNormalization layer with input shape
#         model.add(BatchNormalization(input_shape=(n_inputs,)))

#         activation_settings = DotDict({
#             "elu": ("elu", "he_uniform", "Dropout"),
#             "relu": ("relu", "he_uniform", "Dropout"),
#             # "prelu": ("PReLU", "he_normal", "Dropout"),
#             "selu": ("selu", "lecun_normal", "AlphaDropout"),
#             "tanh": ("tanh", "glorot_normal", "Dropout"),
#             "softmax": ("softmax", "glorot_normal", "Dropout"),
#         })
#         keras_act_name, init_name, dropout_layer = activation_settings[self.activation]

#         # hidden layers
#         for n_nodes in self.layers:
#             model.add(Dense(
#                 units=n_nodes,
#                 activation=keras_act_name,
#             ))

#             # Potentially add dropout layer after each hidden layer
#             if self.dropout:
#                 Dropout = getattr(keras.layers, dropout_layer)
#                 model.add(Dropout(self.dropout))

#         # output layer
#         model.add(Dense(n_outputs, activation="softmax"))

#         # compile the network
#         optimizer = keras.optimizers.Adam(
#             learning_rate=self.learningrate, beta_1=0.9, beta_2=0.999,
#             epsilon=1e-6, amsgrad=False,
#         )

#         model_compile_kwargs = {
#             "loss": "categorical_crossentropy" if self.negative_weights == "ignore" else cumulated_crossentropy,
#             "optimizer": optimizer,
#             "metrics": ["categorical_accuracy"],
#             "weighted_metrics": ["categorical_accuracy"],
#         }
#         if self.loss == "focal_loss":
#             from keras.losses import CategoricalFocalCrossentropy
#             model_compile_kwargs["loss"] = CategoricalFocalCrossentropy(
#                 alpha=self.focal_loss_alpha,
#                 gamma=self.focal_loss_gamma,
#             )
#         model.compile(**model_compile_kwargs)

#         return model


# class CallbacksBase(object):
#     """ Base class that handles parametrization of callbacks """
#     _default__callbacks: set[str] = {
#         "backup", "checkpoint", "reduce_lr",
#         # "early_stopping",
#     }
#     remove_backup: bool = True

#     # NOTE: we could remove these parameters since they can be implemented via reduce_lr_kwargs
#     _default__reduce_lr_factor: float = 0.8
#     _default__reduce_lr_patience: int = 3

#     # custom callback kwargs
#     checkpoint_kwargs: dict = {}
#     backup_kwargs: dict = {}
#     early_stopping_kwargs: dict = {}
#     reduce_lr_kwargs: dict = {}

#     def cast_ml_param_values(self):
#         """
#         Cast the values of the parameters to the correct types
#         """
#         super().cast_ml_param_values()
#         self.callbacks = set(self.callbacks)
#         self.remove_backup = bool(self.remove_backup)
#         self.reduce_lr_factor = float(self.reduce_lr_factor)
#         self.reduce_lr_patience = int(self.reduce_lr_patience)

#     def get_callbacks(self, output):
#         import tensorflow.keras as keras
#         # check that only valid options have been requested
#         callback_options = {"backup", "checkpoint", "reduce_lr", "early_stopping"}
#         if diff := self.callbacks.difference(callback_options):
#             logger.warning(f"Callbacks '{diff}' have been requested but are not properly implemented")

#         # list of callbacks to be returned at the end
#         callbacks = []

#         # output used for BackupAndRestore callback (not deleted by --remove-output)
#         # NOTE: does that work when running remote?
#         # TODO: we should also save the parameters + input_features in the backup to ensure that they
#         #       are equivalent (delete backup if not)
#         backup_output = output["mlmodel"].sibling(f"backup_{output['mlmodel'].basename}", type="d")
#         if self.remove_backup:
#             backup_output.remove()

#         #
#         # for each requested callback, merge default kwargs with custom callback kwargs
#         #

#         if "backup" in self.callbacks:
#             backup_kwargs = dict(
#                 backup_dir=backup_output.abspath,
#             )
#             backup_kwargs.update(self.backup_kwargs)
#             callbacks.append(keras.callbacks.BackupAndRestore(**backup_kwargs))

#         if "checkpoint" in self.callbacks:
#             checkpoint_kwargs = dict(
#                 filepath=output["checkpoint"].abspath,
#                 save_weights_only=False,
#                 monitor="val_loss",
#                 mode="auto",
#                 save_best_only=True,
#             )
#             checkpoint_kwargs.update(self.checkpoint_kwargs)
#             callbacks.append(keras.callbacks.ModelCheckpoint(**checkpoint_kwargs))

#         if "early_stopping" in self.callbacks:
#             early_stopping_kwargs = dict(
#                 monitor="val_loss",
#                 min_delta=0,
#                 patience=max(min(50, int(self.epochs / 5)), 10),
#                 verbose=1,
#                 restore_best_weights=True,
#                 start_from_epoch=max(min(50, int(self.epochs / 5)), 10),
#             )
#             early_stopping_kwargs.update(self.early_stopping_kwargs)
#             callbacks.append(keras.callbacks.EarlyStopping(**early_stopping_kwargs))

#         if "reduce_lr" in self.callbacks:
#             reduce_lr_kwargs = dict(
#                 monitor="val_loss",
#                 factor=self.reduce_lr_factor,
#                 patience=self.reduce_lr_patience,
#                 verbose=1,
#                 mode="auto",
#                 min_delta=0,
#                 min_lr=0,
#             )
#             reduce_lr_kwargs.update(self.reduce_lr_kwargs)
#             callbacks.append(keras.callbacks.ReduceLROnPlateau(**reduce_lr_kwargs))

#         if len(callbacks) != len(self.callbacks):
#             logger.warning(
#                 f"{len(self.callbacks)} callbacks have been requested but only {len(callbacks)} are returned",
#             )

#         return callbacks


# class ClassicModelFitMixin(CallbacksBase):
#     """
#     Mixin to run ML Training with "classic" training loop.
#     TODO: this will require a different reweighting
#     """

#     _default__callbacks: set = {
#         "backup", "checkpoint", "reduce_lr",
#         # "early_stopping",
#     }
#     remove_backup: bool = True
#     _default__reduce_lr_factor: float = 0.8
#     _default__reduce_lr_patience: int = 3
#     _default__epochs: int = 200
#     _default__batchsize: int = 2 ** 12

#     def cast_ml_param_values(self):
#         """
#         Cast the values of the parameters to the correct types
#         """
#         super().cast_ml_param_values()
#         self.epochs = int(self.epochs)
#         self.batchsize = int(self.batchsize)

#     def fit_ml_model(
#         self,
#         task: law.Task,
#         model,
#         train: DotDict[np.array],
#         validation: DotDict[np.array],
#         output,
#     ) -> None:
#         """
#         Training loop with normal tf dataset
#         """
#         import tensorflow as tf

#         log_memory("start")

#         tf_train = tf.data.Dataset.from_tensor_slices(
#             (train["inputs"], train["target"], train["weights"]),
#         ).batch(self.batchsize).prefetch(tf.data.AUTOTUNE)
#         tf_validation = tf.data.Dataset.from_tensor_slices(
#             (validation["inputs"], validation["target"], validation["weights"]),
#         ).batch(self.batchsize).prefetch(tf.data.AUTOTUNE)

#         log_memory("init")

#         # set the kwargs used for training
#         model_fit_kwargs = {
#             "validation_data": tf_validation,
#             "epochs": self.epochs,
#             "verbose": 2,
#             "callbacks": self.get_callbacks(output),
#         }

#         logger.info("Starting training...")
#         tf.debugging.set_log_device_placement(True)
#         model.fit(
#             tf_train,
#             **model_fit_kwargs,
#         )
#         log_memory("loop")

#         # delete tf datasets to clear memory
#         del tf_train
#         del tf_validation
#         log_memory("del")


# class ModelFitMixin(CallbacksBase):
#     # parameters related to callbacks
#     _default__callbacks: set = {
#         "backup", "checkpoint", "reduce_lr",
#         # "early_stopping",
#     }
#     remove_backup: bool = True
#     _default__reduce_lr_factor: float = 0.8
#     _default__reduce_lr_patience: int = 3

#     _default__epochs: int = 200
#     _default__batchsize: int = 2 ** 12
#     # either set steps directly or use attribute from the MultiDataset
#     steps_per_epoch: Union[int, str] = "iter_smallest_process"

#     def cast_ml_param_values(self):
#         """
#         Cast the values of the parameters to the correct types
#         """
#         super().cast_ml_param_values()
#         self.epochs = int(self.epochs)
#         self.batchsize = int(self.batchsize)
#         if isinstance(self.steps_per_epoch, float):
#             self.steps_per_epoch = int(self.steps_per_epoch)
#         else:
#             self.steps_per_epoch = str(self.steps_per_epoch)

#     def resolve_weights_xsec(self, data, max_diff_int: float = 0.3):
#         """
#         Represents cross-section weighting
#         """
#         rel_sumw_dict = {proc_inst: {} for proc_inst in data.keys()}
#         factor = 1
#         smallest_sumw = None
#         for proc_inst, arrays in data.items():
#             sumw = np.sum(arrays.weights) * proc_inst.x.sub_process_class_factor
#             if not smallest_sumw or smallest_sumw >= sumw:
#                 smallest_sumw = sumw

#         for proc_inst, arrays in data.items():
#             sumw = np.sum(arrays.weights) * proc_inst.x.sub_process_class_factor
#             rel_sumw = sumw / smallest_sumw
#             rel_sumw_dict[proc_inst] = rel_sumw

#             if (rel_sumw - round(rel_sumw)) / rel_sumw > max_diff_int:
#                 factor = 2

#         rel_sumw_dict = {proc_inst: int(rel_sumw * factor) for proc_inst, rel_sumw in rel_sumw_dict.items()}

#         return rel_sumw_dict

#     def get_batch_sizes(self, data, round: str = "down"):
#         batch_sizes = {}
#         rel_sumw_dicts = {}
#         for train_node_proc_name, node_config in self.train_nodes.items():
#             train_node_proc = self.config_inst.get_process(train_node_proc_name)
#             sub_procs = (
#                 (set(node_config.get("sub_processes", set())) | {train_node_proc_name}) &
#                 {proc.name for proc in data.keys()}
#             )
#             if not sub_procs:
#                 raise ValueError(f"Cannot find any sub-processes for {train_node_proc_name} in the data")
#             sub_procs = {self.config_inst.get_process(proc_name) for proc_name in sub_procs}
#             class_factor_mode = train_node_proc.x("class_factor_mode", "equal")
#             if class_factor_mode == "xsec":
#                 rel_sumw_dicts[train_node_proc.name] = self.resolve_weights_xsec(
#                     {proc_inst: data[proc_inst] for proc_inst in sub_procs},
#                 )
#             elif class_factor_mode == "equal":
#                 rel_sumw_dicts[train_node_proc.name] = {
#                     proc_inst: proc_inst.x("sub_process_class_factor", 1) for proc_inst in sub_procs
#                 }
#         for train_node_proc_name, node_config in self.train_nodes.items():
#             train_node_proc = self.config_inst.get_process(train_node_proc_name)
#             rel_sumw = rel_sumw_dicts[train_node_proc.name]
#             rel_sumw_dicts[train_node_proc.name]["sum"] = sum([rel_sumw[proc_inst] for proc_inst in rel_sumw.keys()])
#             rel_sumw_dicts[train_node_proc.name]["min"] = min([rel_sumw[proc_inst] for proc_inst in rel_sumw.keys()])

#         lcm_list = lambda numbers: functools.reduce(math.lcm, numbers)
#         lcm = lcm_list([rel_sumw["sum"] for rel_sumw in rel_sumw_dicts.values()])

#         for train_node_proc_name, node_config in self.train_nodes.items():
#             train_node_proc = self.config_inst.get_process(train_node_proc_name)
#             class_factor = self.class_factors.get(train_node_proc.name, 1)
#             rel_sumw_dict = rel_sumw_dicts[train_node_proc.name]
#             batch_factor = class_factor * lcm // rel_sumw_dict["sum"]
#             if not isinstance(batch_factor, int):
#                 raise ValueError(
#                     f"Batch factor {batch_factor} is not an integer. "
#                     "This is likely due to a non-integer class factor.",
#                 )

#             for proc_inst, rel_sumw in rel_sumw_dict.items():
#                 if isinstance(proc_inst, str):
#                     continue
#                 batch_sizes[proc_inst] = rel_sumw * batch_factor

#         # if we requested a batchsize, scale the batch sizes to the requested batchsize, rounding up or down
#         if not self.batchsize:
#             return batch_sizes
#         elif round == "down":
#             batch_scaler = self.batchsize // sum(batch_sizes.values()) or 1
#         elif round == "up":
#             batch_scaler = math.ceil(self.batchsize / sum(batch_sizes.values()))
#         else:
#             raise ValueError(f"Unknown round option {round}")

#         batch_sizes = {proc_inst: int(batch_size * batch_scaler) for proc_inst, batch_size in batch_sizes.items()}
#         return batch_sizes

#     def set_validation_weights(self, validation, batch_sizes, train_steps_per_epoch):
#         """
#         Update the train weights such that
#         """
#         for proc_inst, arrays in validation.items():
#             bs = batch_sizes[proc_inst]
#             arrays.validation_weights = arrays.weights / np.sum(arrays.weights) * bs * train_steps_per_epoch

#     def _check_weights(self, train):
#         sum_nodes = np.zeros(len(self.train_nodes), dtype=np.float32)
#         for proc, data in train.items():
#             sum_nodes += np.bincount(data.labels, weights=data.train_weights, minlength=len(self.train_nodes))
#             logger.info(f"Sum of weights for process {proc}: {np.sum(data.train_weights)}")

#         for proc, node_config in self.train_nodes.items():
#             logger.info(f"Sum of weights for train node process {proc}: {sum_nodes[node_config['ml_id']]}")

#     @timeit
#     def fit_ml_model(
#         self,
#         task: law.Task,
#         model,
#         train: DotDict[np.array],
#         validation: DotDict[np.array],
#         output,
#     ) -> None:
#         """
#         Training loop but with custom dataset
#         """
#         # import tensorflow as tf
#         from hbw.ml.tf_util import MultiDataset
#         from hbw.ml.plotting import plot_history

#         log_memory("start")

#         batch_sizes = self.get_batch_sizes(data=train)
#         print("batch_sizes:", batch_sizes)

#         # Create MultiDataset
#         tf_train = MultiDataset(data=train, batch_size=batch_sizes, kind="train", buffersize=0)
#         log_memory("tf_train")

#         # cleanup memory (TODO: seems to not do much if anything)
#         for key, ml_dataset in train.items():
#             ml_dataset.cleanup()
#         log_memory("train cleanup")

#         # determine the requested steps_per_epoch
#         if isinstance(self.steps_per_epoch, str):
#             magic_smooth_factor = 1
#             # steps_per_epoch is usually "iter_smallest_process" (TODO: check performance with other factors)
#             steps_per_epoch = getattr(tf_train, self.steps_per_epoch) * magic_smooth_factor
#         else:
#             raise Exception("self.steps_per_epoch is not a string, cannot determine steps_per_epoch")
#         if not isinstance(steps_per_epoch, int):
#             raise Exception(
#                 f"steps_per_epoch is {self.steps_per_epoch} but has to be either an integer or"
#                 "a string corresponding to an integer attribute of the MultiDataset",
#             )
#         logger.info(f"Training will be done with {steps_per_epoch} steps per epoch")

#         # Create validation dataset
#         self.set_validation_weights(validation, batch_sizes, steps_per_epoch)
#         tf_validation = MultiDataset(data=validation, kind="valid", buffersize=0)
#         log_memory("tf_validation")

#         for key, ml_dataset in validation.items():
#             ml_dataset.cleanup()
#         log_memory("validation cleanup")

#         # check that the weights are set correctly
#         # self._check_weights(train)

#         # set the kwargs used for training
#         model_fit_kwargs = {
#             "validation_data": (x for x in tf_validation),
#             "validation_steps": tf_validation.iter_smallest_process,
#             "epochs": self.epochs,
#             "verbose": 2,
#             "steps_per_epoch": steps_per_epoch,
#             "callbacks": self.get_callbacks(output),
#         }
#         # start training by iterating over the MultiDataset
#         iterator = (x for x in tf_train)
#         logger.info("Starting training...")
#         # Removed debugger call for performance
#         model.fit(
#             iterator,
#             **model_fit_kwargs,
#         )

#         # Explicit cleanup to prevent memory leaks
#         try:
#             tf_train.cleanup_resources()
#             tf_validation.cleanup_resources()
#         except AttributeError:
#             # Fallback if cleanup_resources doesn't exist
#             pass

#         # Delete references to datasets
#         del tf_train
#         del tf_validation
#         del iterator

#         # create history plots
#         for metric, ylabel, yscale in (
#             ("loss", "Loss", "log"),
#             ("categorical_accuracy", "Accuracy", "linear"),
#             ("weighted_categorical_accuracy", "Weighted Accuracy", "linear"),
#         ):
#             call_func_safe(
#                 plot_history,
#                 model.history.history,
#                 output["plots"],
#                 metric=metric,
#                 ylabel=ylabel,
#                 yscale=yscale,
#             )

#         # Force garbage collection
#         import gc
#         gc.collect()

#         log_memory("cleanup")

# coding: utf-8
 
"""
Mixin classes to build ML models
"""
 
from __future__ import annotations
 
import functools
import math
 
import law
# import order as od
 
from columnflow.types import Union
from columnflow.util import maybe_import, DotDict
from hbw.util import log_memory, call_func_safe, timeit
 
 
np = maybe_import("numpy")
ak = maybe_import("awkward")
 
logger = law.logger.get_logger(__name__)
 
 
def loop_dataset(data, max_count=10000):
    for i, x in enumerate(data):
        if i % int(max_count / 100) == 0:
            print(i)
        if i == max_count:
            break
 
 
class DenseModelMixin(object):
    """
    Mixin that provides an implementation for `prepare_ml_model`
    """
 
    _default__activation: str = "relu"
    _default__layers: tuple[int] = (64, 64, 64)
    _default__dropout: float = 0.50
    _default__learningrate: float = 0.00050
 
    # L2 regularization strength applied to all Dense layer kernels.
    # WHY: tthh_4b has only 3621 samples but is repeated ~22x/epoch via
    # iter_largest_process, so each sample is seen ~4400 times over 200 epochs.
    # Dropout at 0.5 helps, but L2 adds a constant weight-magnitude penalty
    # that further prevents the model from carving sharp decision boundaries
    # around memorized minority-class events.
    # Tuning guide:
    #   1e-4  -> default starting point (gentle, try first)
    #   1e-3  -> stronger, use if tthh_4b train accuracy stays > 0.85
    #   1e-2  -> very strong, only if severe memorization persists
    _default__l2_regularization: float = 1e-3
 
    # Loss: focal_loss preferred over categorical_crossentropy for imbalanced classes.
    loss: str = "focal_loss"
    focal_loss_alpha: float = 0.25
    # gamma=2.0 is the standard focal loss value. Previously tried 3.0 but
    # combined with iter_largest_process oversampling it over-penalised majority
    # classes. Keep at 2.0; the oversampling already handles hard examples.
    focal_loss_gamma: float = 2.0
 
    def cast_ml_param_values(self):
        """
        Cast the values of the parameters to the correct types
        """
        super().cast_ml_param_values()
        self.activation = str(self.activation)
        self.layers = tuple(int(n_nodes) for n_nodes in self.layers)
        self.dropout = float(self.dropout)
        self.learningrate = float(self.learningrate)
        self.l2_regularization = float(self.l2_regularization)
 
        self.loss = str(self.loss)
        self.focal_loss_alpha = float(self.focal_loss_alpha)
        self.focal_loss_gamma = float(self.focal_loss_gamma)
 
    def prepare_ml_model(
        self,
        task: law.Task,
    ):
        import tensorflow.keras as keras
        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization
        from hbw.ml.tf_util import cumulated_crossentropy
 
        n_inputs = len(set(self.input_features))
        n_outputs = len(self.train_nodes.keys())
 
        # define the DNN model
        model = Sequential()
 
        # BatchNormalization layer with input shape
        model.add(BatchNormalization(input_shape=(n_inputs,)))
 
        activation_settings = DotDict({
            "elu": ("elu", "he_uniform", "Dropout"),
            "relu": ("relu", "he_uniform", "Dropout"),
            "selu": ("selu", "lecun_normal", "AlphaDropout"),
            "tanh": ("tanh", "glorot_normal", "Dropout"),
            "softmax": ("softmax", "glorot_normal", "Dropout"),
        })
        keras_act_name, init_name, dropout_layer = activation_settings[self.activation]
 
        # hidden layers
        # L2 regularization is applied to every Dense layer kernel to prevent
        # tthh_4b memorization. With only 3621 samples seen ~4400 times each,
        # large weights can form that are very specific to training events.
        l2_reg = keras.regularizers.l2(self.l2_regularization) if self.l2_regularization else None
        for n_nodes in self.layers:
            model.add(Dense(
                units=n_nodes,
                activation=keras_act_name,
                kernel_regularizer=l2_reg,
            ))
 
            # Potentially add dropout layer after each hidden layer
            if self.dropout:
                Dropout = getattr(keras.layers, dropout_layer)
                model.add(Dropout(self.dropout))
 
        # output layer
        model.add(Dense(n_outputs, activation="softmax"))
 
        # compile the network
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learningrate, beta_1=0.9, beta_2=0.999,
            epsilon=1e-6, amsgrad=False,
        )
 
        model_compile_kwargs = {
            "loss": "categorical_crossentropy" if self.negative_weights == "ignore" else cumulated_crossentropy,
            "optimizer": optimizer,
            "metrics": ["categorical_accuracy"],
            "weighted_metrics": ["categorical_accuracy"],
        }
        if self.loss == "focal_loss":
            from keras.losses import CategoricalFocalCrossentropy
            model_compile_kwargs["loss"] = CategoricalFocalCrossentropy(
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
            )
        model.compile(**model_compile_kwargs)
 
        return model
 
 
class CallbacksBase(object):
    """ Base class that handles parametrization of callbacks """
 
    # FIX 3: early_stopping is now enabled by default.
    # It monitors val_loss and restores the best weights, preventing tthh_4b
    # from continuing to overfit after its optimal validation point.
    _default__callbacks: set[str] = {
        "backup", "checkpoint", "reduce_lr", "early_stopping",
    }
    remove_backup: bool = True
 
    _default__reduce_lr_factor: float = 0.8
    _default__reduce_lr_patience: int = 3
 
    # custom callback kwargs
    checkpoint_kwargs: dict = {}
    backup_kwargs: dict = {}
    early_stopping_kwargs: dict = {}
    reduce_lr_kwargs: dict = {}
 
    def cast_ml_param_values(self):
        """
        Cast the values of the parameters to the correct types
        """
        super().cast_ml_param_values()
        self.callbacks = set(self.callbacks)
        self.remove_backup = bool(self.remove_backup)
        self.reduce_lr_factor = float(self.reduce_lr_factor)
        self.reduce_lr_patience = int(self.reduce_lr_patience)
 
    def get_callbacks(self, output):
        import tensorflow.keras as keras
        # check that only valid options have been requested
        callback_options = {"backup", "checkpoint", "reduce_lr", "early_stopping"}
        if diff := self.callbacks.difference(callback_options):
            logger.warning(f"Callbacks '{diff}' have been requested but are not properly implemented")
 
        callbacks = []
 
        backup_output = output["mlmodel"].sibling(f"backup_{output['mlmodel'].basename}", type="d")
        if self.remove_backup:
            backup_output.remove()
 
        if "backup" in self.callbacks:
            backup_kwargs = dict(
                backup_dir=backup_output.abspath,
            )
            backup_kwargs.update(self.backup_kwargs)
            callbacks.append(keras.callbacks.BackupAndRestore(**backup_kwargs))
 
        if "checkpoint" in self.callbacks:
            checkpoint_kwargs = dict(
                filepath=output["checkpoint"].abspath,
                save_weights_only=False,
                monitor="val_loss",
                mode="auto",
                save_best_only=True,
            )
            checkpoint_kwargs.update(self.checkpoint_kwargs)
            callbacks.append(keras.callbacks.ModelCheckpoint(**checkpoint_kwargs))
 
        if "early_stopping" in self.callbacks:
            early_stopping_kwargs = dict(
                monitor="val_loss",
                min_delta=0,
                # FIX 3b: patience scaled to epochs but bounded. With 200 epochs,
                # patience=40 gives the model enough time to escape local minima
                # while preventing long overfit runs on tthh_4b.
                patience=max(min(50, int(self.epochs / 5)), 10),
                verbose=1,
                restore_best_weights=True,
                start_from_epoch=max(min(50, int(self.epochs / 5)), 10),
            )
            early_stopping_kwargs.update(self.early_stopping_kwargs)
            callbacks.append(keras.callbacks.EarlyStopping(**early_stopping_kwargs))
 
        if "reduce_lr" in self.callbacks:
            reduce_lr_kwargs = dict(
                monitor="val_loss",
                factor=self.reduce_lr_factor,
                patience=self.reduce_lr_patience,
                verbose=1,
                mode="auto",
                min_delta=0,
                min_lr=0,
            )
            reduce_lr_kwargs.update(self.reduce_lr_kwargs)
            callbacks.append(keras.callbacks.ReduceLROnPlateau(**reduce_lr_kwargs))
 
        if len(callbacks) != len(self.callbacks):
            logger.warning(
                f"{len(self.callbacks)} callbacks have been requested but only {len(callbacks)} are returned",
            )
 
        return callbacks
 
 
class ClassicModelFitMixin(CallbacksBase):
    """
    Mixin to run ML Training with "classic" training loop.
    """
 
    _default__callbacks: set = {
        "backup", "checkpoint", "reduce_lr", "early_stopping",
    }
    remove_backup: bool = True
    _default__reduce_lr_factor: float = 0.8
    _default__reduce_lr_patience: int = 3
    _default__epochs: int = 200
    _default__batchsize: int = 2 ** 12
 
    def cast_ml_param_values(self):
        super().cast_ml_param_values()
        self.epochs = int(self.epochs)
        self.batchsize = int(self.batchsize)
 
    def compute_class_weights(self, train):
        """
        FIX 4: Compute inverse-frequency class weights from training labels.
        Classes with fewer samples (e.g. tthh_4b with ~3.6k vs HHH with ~90k)
        receive proportionally higher weights in the loss, forcing the model to
        treat each class equally regardless of sample count.
 
        Returns a dict {class_index: weight} suitable for Keras class_weight arg.
        """
        n_classes = len(self.train_nodes)
        all_labels = np.concatenate([arrays.labels for arrays in train.values()])
        class_counts = np.bincount(all_labels, minlength=n_classes).astype(float)
        total = len(all_labels)
 
        class_weights = {}
        for i, count in enumerate(class_counts):
            class_weights[i] = (total / (n_classes * count)) if count > 0 else 1.0
 
        for node_name, node_config in self.train_nodes.items():
            ml_id = node_config["ml_id"]
            logger.info(
                f"Class weight for {node_name} (id={ml_id}): "
                f"{class_weights[ml_id]:.4f} "
                f"(n_samples={int(class_counts[ml_id])})"
            )
        return class_weights
 
    def fit_ml_model(
        self,
        task: law.Task,
        model,
        train: DotDict[np.array],
        validation: DotDict[np.array],
        output,
    ) -> None:
        import tensorflow as tf
 
        log_memory("start")
 
        tf_train = tf.data.Dataset.from_tensor_slices(
            (train["inputs"], train["target"], train["weights"]),
        ).batch(self.batchsize).prefetch(tf.data.AUTOTUNE)
        tf_validation = tf.data.Dataset.from_tensor_slices(
            (validation["inputs"], validation["target"], validation["weights"]),
        ).batch(self.batchsize).prefetch(tf.data.AUTOTUNE)
 
        log_memory("init")
 
        # FIX 4: compute and log class weights before training
        class_weights = self.compute_class_weights(train)
 
        model_fit_kwargs = {
            "validation_data": tf_validation,
            "epochs": self.epochs,
            "verbose": 2,
            "callbacks": self.get_callbacks(output),
            # FIX 4: pass class weights so minority classes are upweighted in loss
            "class_weight": class_weights,
        }
 
        logger.info("Starting training...")
        tf.debugging.set_log_device_placement(True)
        model.fit(tf_train, **model_fit_kwargs)
        log_memory("loop")
 
        del tf_train
        del tf_validation
        log_memory("del")
 
 
class ModelFitMixin(CallbacksBase):
    _default__callbacks: set = {
        "backup", "checkpoint", "reduce_lr", "early_stopping",
    }
    remove_backup: bool = True
    _default__reduce_lr_factor: float = 0.8
    _default__reduce_lr_patience: int = 3
 
    _default__epochs: int = 200
    _default__batchsize: int = 2 ** 12
 
    # FIX 5: Changed from "iter_smallest_process" to "iter_largest_process".
    # Previously, with tthh_4b having only 3 steps/epoch, training terminated
    # after just 3 gradient updates — the model memorised the tiny tthh_4b set
    # (train=0.88) but never generalised (val=0.52).
    # Using iter_largest_process (66 steps from HHH) means tthh_4b is cycled
    # through ~22x per epoch, acting as natural oversampling and exposing the
    # model to enough tthh_4b examples to learn generalisable features.
    steps_per_epoch: Union[int, str] = "iter_largest_process"
 
    def cast_ml_param_values(self):
        super().cast_ml_param_values()
        self.epochs = int(self.epochs)
        self.batchsize = int(self.batchsize)
        if isinstance(self.steps_per_epoch, float):
            self.steps_per_epoch = int(self.steps_per_epoch)
        else:
            self.steps_per_epoch = str(self.steps_per_epoch)
 
    def resolve_weights_xsec(self, data, max_diff_int: float = 0.3):
        """
        Represents cross-section weighting
        """
        rel_sumw_dict = {proc_inst: {} for proc_inst in data.keys()}
        factor = 1
        smallest_sumw = None
        for proc_inst, arrays in data.items():
            sumw = np.sum(arrays.weights) * proc_inst.x.sub_process_class_factor
            if not smallest_sumw or smallest_sumw >= sumw:
                smallest_sumw = sumw
 
        for proc_inst, arrays in data.items():
            sumw = np.sum(arrays.weights) * proc_inst.x.sub_process_class_factor
            rel_sumw = sumw / smallest_sumw
            rel_sumw_dict[proc_inst] = rel_sumw
 
            if (rel_sumw - round(rel_sumw)) / rel_sumw > max_diff_int:
                factor = 2
 
        rel_sumw_dict = {proc_inst: int(rel_sumw * factor) for proc_inst, rel_sumw in rel_sumw_dict.items()}
 
        return rel_sumw_dict
 
    def get_batch_sizes(self, data, round: str = "down"):
        batch_sizes = {}
        rel_sumw_dicts = {}
        for train_node_proc_name, node_config in self.train_nodes.items():
            train_node_proc = self.config_inst.get_process(train_node_proc_name)
            sub_procs = (
                (set(node_config.get("sub_processes", set())) | {train_node_proc_name}) &
                {proc.name for proc in data.keys()}
            )
            if not sub_procs:
                raise ValueError(f"Cannot find any sub-processes for {train_node_proc_name} in the data")
            sub_procs = {self.config_inst.get_process(proc_name) for proc_name in sub_procs}
            class_factor_mode = train_node_proc.x("class_factor_mode", "equal")
            if class_factor_mode == "xsec":
                rel_sumw_dicts[train_node_proc.name] = self.resolve_weights_xsec(
                    {proc_inst: data[proc_inst] for proc_inst in sub_procs},
                )
            elif class_factor_mode == "equal":
                rel_sumw_dicts[train_node_proc.name] = {
                    proc_inst: proc_inst.x("sub_process_class_factor", 1) for proc_inst in sub_procs
                }
        for train_node_proc_name, node_config in self.train_nodes.items():
            train_node_proc = self.config_inst.get_process(train_node_proc_name)
            rel_sumw = rel_sumw_dicts[train_node_proc.name]
            rel_sumw_dicts[train_node_proc.name]["sum"] = sum([rel_sumw[proc_inst] for proc_inst in rel_sumw.keys()])
            rel_sumw_dicts[train_node_proc.name]["min"] = min([rel_sumw[proc_inst] for proc_inst in rel_sumw.keys()])
 
        lcm_list = lambda numbers: functools.reduce(math.lcm, numbers)
        lcm = lcm_list([rel_sumw["sum"] for rel_sumw in rel_sumw_dicts.values()])
 
        for train_node_proc_name, node_config in self.train_nodes.items():
            train_node_proc = self.config_inst.get_process(train_node_proc_name)
            class_factor = self.class_factors.get(train_node_proc.name, 1)
            rel_sumw_dict = rel_sumw_dicts[train_node_proc.name]
            batch_factor = class_factor * lcm // rel_sumw_dict["sum"]
            if not isinstance(batch_factor, int):
                raise ValueError(
                    f"Batch factor {batch_factor} is not an integer. "
                    "This is likely due to a non-integer class factor.",
                )
 
            for proc_inst, rel_sumw in rel_sumw_dict.items():
                if isinstance(proc_inst, str):
                    continue
                batch_sizes[proc_inst] = rel_sumw * batch_factor
 
        if not self.batchsize:
            return batch_sizes
        elif round == "down":
            batch_scaler = self.batchsize // sum(batch_sizes.values()) or 1
        elif round == "up":
            batch_scaler = math.ceil(self.batchsize / sum(batch_sizes.values()))
        else:
            raise ValueError(f"Unknown round option {round}")
 
        batch_sizes = {proc_inst: int(batch_size * batch_scaler) for proc_inst, batch_size in batch_sizes.items()}
        return batch_sizes
 
    def set_validation_weights(self, validation, batch_sizes, train_steps_per_epoch):
        for proc_inst, arrays in validation.items():
            bs = batch_sizes[proc_inst]
            arrays.validation_weights = arrays.weights / np.sum(arrays.weights) * bs * train_steps_per_epoch
 
    def _check_weights(self, train):
        sum_nodes = np.zeros(len(self.train_nodes), dtype=np.float32)
        for proc, data in train.items():
            sum_nodes += np.bincount(data.labels, weights=data.train_weights, minlength=len(self.train_nodes))
            logger.info(f"Sum of weights for process {proc}: {np.sum(data.train_weights)}")
 
        for proc, node_config in self.train_nodes.items():
            logger.info(f"Sum of weights for train node process {proc}: {sum_nodes[node_config['ml_id']]}")
 
    def compute_class_weights(self, train):
        """
        Compute soft sqrt-based class weights from training labels.
 
        WHY SQRT AND NOT FULL INVERSE-FREQUENCY:
        iter_largest_process already oversamples tthh_4b ~22x per epoch.
        Stacking full inverse-frequency weights (14.8x for tthh_4b) on top
        creates a combined 22 × 14.8 ≈ 326x gradient dominance — this caused
        the model to predict tthh_4b for almost everything (HHH accuracy 0.67→0.30).
 
        sqrt(inverse_frequency) gives a gentler correction:
          tthh_4b: sqrt(14.8) ≈ 3.8x   (was 14.8x)
          HHH:     sqrt(0.60) ≈ 0.77x  (was 0.60x)
        Combined with 22x oversampling: 22 × 3.8 ≈ 84x — still strongly favours
        tthh_4b but no longer crushes the other classes.
 
        Tune `class_weight_power` (default 0.5) if needed:
          0.0 = no class weighting (iter_largest_process alone)
          0.5 = sqrt (recommended starting point)
          1.0 = full inverse-frequency (too aggressive with oversampling)
        """
        # class_weight_power = 0.0: iter_largest_process already oversamples
        # tthh_4b ~22x/epoch, which is sufficient to balance gradient exposure.
        # Adding class weights on top (even sqrt-based) compounded the effect
        # and caused HHH/tt accuracy to degrade. L2 regularization now handles
        # the memorization problem more directly.
        # Set to 0.25 or 0.5 only if tthh_4b val accuracy plateaus below 0.6
        # after adding L2.
        class_weight_power = getattr(self, "class_weight_power", 0.0)
 
        n_classes = len(self.train_nodes)
        all_labels = np.concatenate([arrays.labels for arrays in train.values()])
        class_counts = np.bincount(all_labels, minlength=n_classes).astype(float)
        total = len(all_labels)
 
        class_weights = {}
        for i, count in enumerate(class_counts):
            raw = (total / (n_classes * count)) if count > 0 else 1.0
            class_weights[i] = raw ** class_weight_power
 
        for node_name, node_config in self.train_nodes.items():
            ml_id = node_config["ml_id"]
            logger.info(
                f"Class weight for {node_name} (id={ml_id}): "
                f"{class_weights[ml_id]:.4f} "
                f"(n_samples={int(class_counts[ml_id])}, power={class_weight_power})"
            )
        return class_weights

    def apply_class_weights_to_samples(self, train, class_weights):
        """
        Bake class weights into per-sample weights in-place, BEFORE the
        MultiDataset is constructed.

        Keras raises ValueError when class_weight is passed with a Python
        generator input. Multiplying sample weights directly is mathematically
        equivalent and works with any input type.
        """
        for proc_inst, arrays in train.items():
            sample_cw = np.array(
                [class_weights[label] for label in arrays.labels],
                dtype=np.float32,
            )
            arrays._train_weights = (arrays.train_weights * sample_cw).astype(np.float32)
            logger.info(
                f"Applied class weights to {proc_inst.name}: "
                f"sum of train_weights {np.sum(arrays.train_weights):.2f}"
            )

    @timeit
    def fit_ml_model(
        self,
        task: law.Task,
        model,
        train: DotDict[np.array],
        validation: DotDict[np.array],
        output,
    ) -> None:
        """
        Training loop with custom MultiDataset, now with oversampling via
        iter_largest_process and class weights to fix class imbalance.
        """
        from hbw.ml.tf_util import MultiDataset
        from hbw.ml.plotting import plot_history
 
        log_memory("start")
 
        # FIX 4: compute and bake class weights into sample weights BEFORE
        # MultiDataset is created — Keras rejects class_weight with generators.
        class_weights = self.compute_class_weights(train)
        self.apply_class_weights_to_samples(train, class_weights)
        log_memory("class_weights applied")
 
        batch_sizes = self.get_batch_sizes(data=train)
        print("batch_sizes:", batch_sizes)
 
        tf_train = MultiDataset(data=train, batch_size=batch_sizes, kind="train", buffersize=0)
        log_memory("tf_train")
 
        for key, ml_dataset in train.items():
            ml_dataset.cleanup()
        log_memory("train cleanup")
 
        # FIX 5: resolve steps_per_epoch from MultiDataset attribute
        if isinstance(self.steps_per_epoch, str):
            magic_smooth_factor = 1
            steps_per_epoch = getattr(tf_train, self.steps_per_epoch, None)
            if steps_per_epoch is None:
                # Graceful fallback: if iter_largest_process is not available on
                # this version of MultiDataset, fall back to iter_smallest_process
                logger.warning(
                    f"MultiDataset has no attribute '{self.steps_per_epoch}', "
                    "falling back to 'iter_smallest_process'. "
                    "Consider updating MultiDataset to support iter_largest_process.",
                )
                steps_per_epoch = tf_train.iter_smallest_process
            steps_per_epoch = steps_per_epoch * magic_smooth_factor
        else:
            raise Exception("self.steps_per_epoch is not a string, cannot determine steps_per_epoch")
        if not isinstance(steps_per_epoch, int):
            raise Exception(
                f"steps_per_epoch is {self.steps_per_epoch} but has to be either an integer or"
                "a string corresponding to an integer attribute of the MultiDataset",
            )
        logger.info(f"Training will be done with {steps_per_epoch} steps per epoch")
 
        self.set_validation_weights(validation, batch_sizes, steps_per_epoch)
        tf_validation = MultiDataset(data=validation, kind="valid", buffersize=0)
        log_memory("tf_validation")
 
        for key, ml_dataset in validation.items():
            ml_dataset.cleanup()
        log_memory("validation cleanup")
 
        model_fit_kwargs = {
            "validation_data": (x for x in tf_validation),
            "validation_steps": tf_validation.iter_smallest_process,
            "epochs": self.epochs,
            "verbose": 2,
            "steps_per_epoch": steps_per_epoch,
            "callbacks": self.get_callbacks(output),
            # NOTE: class_weight is intentionally NOT passed here.
            # Keras raises a ValueError when class_weight is used with a Python
            # generator. Class weighting is instead applied directly to
            # arrays.weights via apply_class_weights_to_samples() above.
        }
 
        iterator = (x for x in tf_train)
        logger.info("Starting training...")
        model.fit(iterator, **model_fit_kwargs)
 
        try:
            tf_train.cleanup_resources()
            tf_validation.cleanup_resources()
        except AttributeError:
            pass
 
        del tf_train
        del tf_validation
        del iterator
 
        for metric, ylabel, yscale in (
            ("loss", "Loss", "log"),
            ("categorical_accuracy", "Accuracy", "linear"),
            ("weighted_categorical_accuracy", "Weighted Accuracy", "linear"),
        ):
            call_func_safe(
                plot_history,
                model.history.history,
                output["plots"],
                metric=metric,
                ylabel=ylabel,
                yscale=yscale,
            )
 
        import gc
        gc.collect()
 
        log_memory("cleanup")
 
