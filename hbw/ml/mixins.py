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

    # TODO: these parameters are currently not part of the MLModel repr
    loss: str = "categorical_crossentropy"
    focal_loss_alpha: float = 0.25
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
        # from keras.losses import CategoricalFocalCrossentropy

        n_inputs = len(set(self.input_features))
        # n_outputs = len(self.processes)
        n_outputs = len(self.train_nodes.keys())

        # define the DNN model
        model = Sequential()

        # BatchNormalization layer with input shape
        model.add(BatchNormalization(input_shape=(n_inputs,)))

        activation_settings = DotDict({
            "elu": ("elu", "he_uniform", "Dropout"),
            "relu": ("relu", "he_uniform", "Dropout"),
            # "prelu": ("PReLU", "he_normal", "Dropout"),
            "selu": ("selu", "lecun_normal", "AlphaDropout"),
            "tanh": ("tanh", "glorot_normal", "Dropout"),
            "softmax": ("softmax", "glorot_normal", "Dropout"),
        })
        keras_act_name, init_name, dropout_layer = activation_settings[self.activation]

        # hidden layers
        for n_nodes in self.layers:
            model.add(Dense(
                units=n_nodes,
                activation=keras_act_name,
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

# class TransformerModelMixin(object):
#     """
#     Mixin that provides an implementation for `prepare_ml_model` using a Transformer architecture.
#     Optimized for tabular HEP classification tasks.
#     """

#     # Transformer-specific hyperparameters
#     _default__embed_dim: int = 64        # Increased: richer feature representations
#     _default__num_heads: int = 4
#     _default__ff_dim: int = 128          # Increased: 2x embed_dim is a good heuristic
#     _default__num_blocks: int = 4        # Increased: deeper = more expressive
#     _default__dropout: float = 0.10
#     _default__learningrate: float = 0.00050
#     _default__mlp_units: int = 128       # NEW: classification head width
#     _default__mlp_layers: int = 2        # NEW: classification head depth

#     loss: str = "categorical_crossentropy"
#     focal_loss_alpha: float = 0.25
#     focal_loss_gamma: float = 2.0

#     def cast_ml_param_values(self):
#         super().cast_ml_param_values()
#         self.embed_dim = int(self.embed_dim)
#         self.num_heads = int(self.num_heads)
#         self.ff_dim = int(self.ff_dim)
#         self.num_blocks = int(self.num_blocks)
#         self.mlp_units = int(self.mlp_units)
#         self.mlp_layers = int(self.mlp_layers)
#         self.dropout = float(self.dropout)
#         self.learningrate = float(self.learningrate)
#         self.loss = str(self.loss)
#         self.focal_loss_alpha = float(self.focal_loss_alpha)
#         self.focal_loss_gamma = float(self.focal_loss_gamma)

#     def prepare_ml_model(self, task: law.Task,):
#         import tensorflow as tf
#         import tensorflow.keras as keras
#         from keras.models import Model
#         from keras.layers import (
#             Input, Dense, BatchNormalization, LayerNormalization,
#             MultiHeadAttention, Dropout, GlobalAveragePooling1D,
#             Reshape, Add, Activation
#         )
#         from hbw.ml.tf_util import cumulated_crossentropy

#         n_inputs = len(set(self.input_features))
#         n_outputs = len(self.train_nodes.keys())

#         # ── 1. Input + Preprocessing ──────────────────────────────────────────
#         inputs = Input(shape=(n_inputs,), name="input")

#         # BatchNorm on raw inputs (mean/variance learned from data)
#         x = BatchNormalization(name="input_norm")(inputs)

#         # Reshape flat features into a "sequence" of tokens: (batch, n_inputs, 1)
#         x = Reshape((n_inputs, 1), name="reshape")(x)

#         # Linear embedding: project each scalar feature → embed_dim
#         # (equivalent to a shared Dense across the sequence dimension)
#         x = Dense(self.embed_dim, use_bias=False, name="feature_embedding")(x)

#         # ── 2. Learnable [CLS] token  ─────────────────────────────────────────
#         # A single trainable token is prepended. After all Transformer blocks
#         # its representation captures a global summary of the full sequence,
#         # which is a cleaner classification signal than average-pooling.
#         batch_size = tf.shape(x)[0]
#         cls_token = self.add_weight_cls_token(n_inputs)  # see helper below
#         cls_tokens = tf.broadcast_to(cls_token, [batch_size, 1, self.embed_dim])
#         x = tf.concat([cls_tokens, x], axis=1)          # (B, n_inputs+1, embed_dim)

#         # ── 3. Transformer Blocks ─────────────────────────────────────────────
#         for i in range(self.num_blocks):
#             # --- Self-Attention sub-layer ---
#             attn_output = MultiHeadAttention(
#                 num_heads=self.num_heads,
#                 key_dim=self.embed_dim // self.num_heads,  # per-head dimension
#                 dropout=self.dropout,
#                 name=f"mha_{i}",
#             )(x, x)
#             attn_output = Dropout(self.dropout, name=f"attn_drop_{i}")(attn_output)
#             x = LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(Add(name=f"res1_{i}")([x, attn_output]))

#             # --- Feed-Forward sub-layer (Pre-LN improves gradient flow) ---
#             ffn = LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x)   # Pre-LN
#             ffn = Dense(self.ff_dim, name=f"ffn_expand_{i}")(ffn)
#             ffn = Activation("gelu", name=f"gelu_{i}")(ffn)              # GELU > ReLU
#             ffn = Dropout(self.dropout, name=f"ffn_drop_{i}")(ffn)
#             ffn = Dense(self.embed_dim, name=f"ffn_contract_{i}")(ffn)
#             ffn = Dropout(self.dropout, name=f"ffn_drop2_{i}")(ffn)
#             x = Add(name=f"res2_{i}")([x, ffn])

#         # ── 4. Readout ────────────────────────────────────────────────────────
#         # Extract CLS token (index 0 along the sequence axis)
#         cls_output = x[:, 0, :]   # (B, embed_dim)

#         # ── 5. Classification Head (MLP) ──────────────────────────────────────
#         # A small MLP after the Transformer head is standard practice and
#         # significantly helps final discrimination power.
#         head = cls_output
#         for j in range(self.mlp_layers):
#             units = max(self.mlp_units // (2 ** j), n_outputs * 2)
#             head = Dense(units, name=f"head_dense_{j}")(head)
#             head = BatchNormalization(name=f"head_bn_{j}")(head)
#             head = Activation("gelu", name=f"head_act_{j}")(head)
#             head = Dropout(self.dropout, name=f"head_drop_{j}")(head)

#         outputs = Dense(n_outputs, activation="softmax", name="output")(head)

#         # ── 6. Build Model ────────────────────────────────────────────────────
#         model = Model(inputs=inputs, outputs=outputs)

#         # ── 7. Compile ────────────────────────────────────────────────────────
#         # Cosine-decay schedule: warm-up prevents early instability,
#         # cosine annealing helps escape sharp minima near the end of training.
#         lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
#             initial_learning_rate=self.learningrate,
#             first_decay_steps=1000,
#             t_mul=2.0,
#             m_mul=0.9,
#         )

#         optimizer = keras.optimizers.AdamW(    # AdamW > Adam for generalisation
#             learning_rate=lr_schedule,
#             weight_decay=1e-4,
#             beta_1=0.9,
#             beta_2=0.999,
#             epsilon=1e-6,
#             amsgrad=False,
#         )

#         loss_fn = (
#             "categorical_crossentropy"
#             if getattr(self, "negative_weights", "") == "ignore"
#             else cumulated_crossentropy
#         )

#         if self.loss == "focal_loss":
#             from keras.losses import CategoricalFocalCrossentropy
#             loss_fn = CategoricalFocalCrossentropy(
#                 alpha=self.focal_loss_alpha,
#                 gamma=self.focal_loss_gamma,
#             )

#         model.compile(
#             loss=loss_fn,
#             optimizer=optimizer,
#             metrics=["categorical_accuracy"],
#             weighted_metrics=["categorical_accuracy"],
#         )

#         return model

#     # ── Helper: CLS token weight ───────────────────────────────────────────────
#     # Must be called on an instance that supports add_weight (e.g. a Keras layer).
#     # If TransformerModelMixin is not itself a Layer, store the token as a
#     # tf.Variable and reference it via closure instead.
#     def add_weight_cls_token(self, n_inputs):
#         import tensorflow as tf
#         if not hasattr(self, "_cls_token"):
#             self._cls_token = tf.Variable(
#                 tf.zeros([1, 1, self.embed_dim]),
#                 trainable=True,
#                 name="cls_token",
#                 dtype=tf.float32,
#             )
#         return self._cls_token


class CallbacksBase(object):
    """ Base class that handles parametrization of callbacks """
    _default__callbacks: set[str] = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True

    # NOTE: we could remove these parameters since they can be implemented via reduce_lr_kwargs
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

        # list of callbacks to be returned at the end
        callbacks = []

        # output used for BackupAndRestore callback (not deleted by --remove-output)
        # NOTE: does that work when running remote?
        # TODO: we should also save the parameters + input_features in the backup to ensure that they
        #       are equivalent (delete backup if not)
        backup_output = output["mlmodel"].sibling(f"backup_{output['mlmodel'].basename}", type="d")
        if self.remove_backup:
            backup_output.remove()

        #
        # for each requested callback, merge default kwargs with custom callback kwargs
        #

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
    TODO: this will require a different reweighting
    """

    _default__callbacks: set = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True
    _default__reduce_lr_factor: float = 0.8
    _default__reduce_lr_patience: int = 3
    _default__epochs: int = 200
    _default__batchsize: int = 2 ** 12

    def cast_ml_param_values(self):
        """
        Cast the values of the parameters to the correct types
        """
        super().cast_ml_param_values()
        self.epochs = int(self.epochs)
        self.batchsize = int(self.batchsize)

    def fit_ml_model(
        self,
        task: law.Task,
        model,
        train: DotDict[np.array],
        validation: DotDict[np.array],
        output,
    ) -> None:
        """
        Training loop with normal tf dataset
        """
        import tensorflow as tf

        log_memory("start")

        tf_train = tf.data.Dataset.from_tensor_slices(
            (train["inputs"], train["target"], train["weights"]),
        ).batch(self.batchsize).prefetch(tf.data.AUTOTUNE)
        tf_validation = tf.data.Dataset.from_tensor_slices(
            (validation["inputs"], validation["target"], validation["weights"]),
        ).batch(self.batchsize).prefetch(tf.data.AUTOTUNE)

        log_memory("init")

        # set the kwargs used for training
        model_fit_kwargs = {
            "validation_data": tf_validation,
            "epochs": self.epochs,
            "verbose": 2,
            "callbacks": self.get_callbacks(output),
        }

        logger.info("Starting training...")
        tf.debugging.set_log_device_placement(True)
        model.fit(
            tf_train,
            **model_fit_kwargs,
        )
        log_memory("loop")

        # delete tf datasets to clear memory
        del tf_train
        del tf_validation
        log_memory("del")


class ModelFitMixin(CallbacksBase):
    # parameters related to callbacks
    _default__callbacks: set = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True
    _default__reduce_lr_factor: float = 0.8
    _default__reduce_lr_patience: int = 3

    _default__epochs: int = 200
    _default__batchsize: int = 2 ** 12
    # either set steps directly or use attribute from the MultiDataset
    steps_per_epoch: Union[int, str] = "iter_smallest_process"

    def cast_ml_param_values(self):
        """
        Cast the values of the parameters to the correct types
        """
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

        # if we requested a batchsize, scale the batch sizes to the requested batchsize, rounding up or down
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
        """
        Update the train weights such that
        """
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
        Training loop but with custom dataset
        """
        # import tensorflow as tf
        from hbw.ml.tf_util import MultiDataset
        from hbw.ml.plotting import plot_history

        log_memory("start")

        batch_sizes = self.get_batch_sizes(data=train)
        print("batch_sizes:", batch_sizes)

        # Create MultiDataset
        tf_train = MultiDataset(data=train, batch_size=batch_sizes, kind="train", buffersize=0)
        log_memory("tf_train")

        # cleanup memory (TODO: seems to not do much if anything)
        for key, ml_dataset in train.items():
            ml_dataset.cleanup()
        log_memory("train cleanup")

        # determine the requested steps_per_epoch
        if isinstance(self.steps_per_epoch, str):
            magic_smooth_factor = 1
            # steps_per_epoch is usually "iter_smallest_process" (TODO: check performance with other factors)
            steps_per_epoch = getattr(tf_train, self.steps_per_epoch) * magic_smooth_factor
        else:
            raise Exception("self.steps_per_epoch is not a string, cannot determine steps_per_epoch")
        if not isinstance(steps_per_epoch, int):
            raise Exception(
                f"steps_per_epoch is {self.steps_per_epoch} but has to be either an integer or"
                "a string corresponding to an integer attribute of the MultiDataset",
            )
        logger.info(f"Training will be done with {steps_per_epoch} steps per epoch")

        # Create validation dataset
        self.set_validation_weights(validation, batch_sizes, steps_per_epoch)
        tf_validation = MultiDataset(data=validation, kind="valid", buffersize=0)
        log_memory("tf_validation")

        for key, ml_dataset in validation.items():
            ml_dataset.cleanup()
        log_memory("validation cleanup")

        # check that the weights are set correctly
        # self._check_weights(train)

        # set the kwargs used for training
        model_fit_kwargs = {
            "validation_data": (x for x in tf_validation),
            "validation_steps": tf_validation.iter_smallest_process,
            "epochs": self.epochs,
            "verbose": 2,
            "steps_per_epoch": steps_per_epoch,
            "callbacks": self.get_callbacks(output),
        }
        # start training by iterating over the MultiDataset
        iterator = (x for x in tf_train)
        logger.info("Starting training...")
        # Removed debugger call for performance
        model.fit(
            iterator,
            **model_fit_kwargs,
        )

        # Explicit cleanup to prevent memory leaks
        try:
            tf_train.cleanup_resources()
            tf_validation.cleanup_resources()
        except AttributeError:
            # Fallback if cleanup_resources doesn't exist
            pass

        # Delete references to datasets
        del tf_train
        del tf_validation
        del iterator

        # create history plots
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

        # Force garbage collection
        import gc
        gc.collect()

        log_memory("cleanup")
