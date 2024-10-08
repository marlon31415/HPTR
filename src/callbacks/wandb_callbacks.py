# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
from pathlib import Path
import wandb
import torch
import torch.nn as nn
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    return None
    # raise Exception("You are using wandb related callback, but WandbLogger was not found for some reason...")


class ModelCheckpointWB(ModelCheckpoint):
    def save_checkpoint(self, trainer) -> None:
        super().save_checkpoint(trainer)
        if not hasattr(self, "_logged_model_time"):
            self._logged_model_time = {}
        logger = get_wandb_logger(trainer)
        if self.current_score is None:
            self.current_score = trainer.callback_metrics.get(self.monitor)
        if logger is not None:
            self._scan_and_log_checkpoints(logger)

    @rank_zero_only
    def _scan_and_log_checkpoints(self, wb_logger: WandbLogger) -> None:
        # adapted from pytorch_lightning 1.4.0: loggers/wandb.py
        checkpoints = {
            self.last_model_path: self.current_score,
            self.best_model_path: self.best_model_score,
        }
        checkpoints = sorted(
            (Path(p).stat().st_mtime, p, s)
            for p, s in checkpoints.items()
            if Path(p).is_file()
        )
        checkpoints = [
            c
            for c in checkpoints
            if c[1] not in self._logged_model_time.keys()
            or self._logged_model_time[c[1]] < c[0]
        ]
        # log iteratively all new checkpoints
        for t, p, s in checkpoints:
            metadata = {
                "score": s.item(),
                "original_filename": Path(p).name,
                "ModelCheckpoint": {
                    k: getattr(self, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                        "_every_n_val_epochs",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(self, k)
                },
            }
            artifact = wandb.Artifact(
                name=wb_logger.experiment.id, type="model", metadata=metadata
            )
            artifact.add_file(p, name="model.ckpt")
            aliases = ["latest", "best"] if p == self.best_model_path else ["latest"]
            wb_logger.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self._log = log
        self._log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer)
        logger.watch(model=trainer.model, log=self._log, log_freq=self._log_freq)


class TracingWrapper(nn.Module):
    def __init__(self, model, input_keys, batch_idx):
        super().__init__()
        self.model = model
        self.input_keys = input_keys
        self.batch_idx = batch_idx

    def forward(self, *inputs):
        # Convert tuple of tensors into a dictionary using input_keys
        input_dict = {key: tensor for key, tensor in zip(self.input_keys, inputs)}
        return self.model(input_dict, self.batch_idx)


class TorchOnesBool(torch.Tensor):
    @staticmethod
    def __new__(cls, size) -> "TorchOnesBool":
        # Create a tensor of ones of type bool
        tensor = super().__new__(cls, torch.ones(*size, dtype=torch.bool))
        return tensor

    def __init__(self, size) -> None:
        pass


class TorchScriptLogger(Callback):
    """Logs a torch.jit.trace model to Weights & Biases."""

    def __init__(self, log_dir: str = "checkpoints/", input_sample=None, batch_idx=0):
        super().__init__()
        self.log_dir = log_dir
        self.input_sample = input_sample
        self.batch_idx = batch_idx

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of training to trace and log the model."""
        logger = get_wandb_logger(trainer)
        if logger is not None:
            # Create the log directory if it doesn't exist
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)

            # Trace the model
            if self.input_sample is None:
                raise ValueError("Input sample for torch.jit.trace is not provided.")

            input_keys = [
                "input/target_valid",
                "input/target_type",
                "input/target_attr",
                "input/other_valid",
                "input/other_attr",
                "input/tl_valid",
                "input/tl_attr",
                "input/map_valid",
                "input/map_attr",
                "ref_pos",
                "ref_rot",
                "agent/valid",
                "agent/pos",
                "agent/vel",
                "agent/spd",
                "agent/acc",
                "agent/yaw_bbox",
                "agent/yaw_rate",
                "agent/type",
                "agent/role",
                "agent/size",
                "agent/cmd",
                "map/valid",
                "map/type",
                "map/pos",
                "map/dir",
                "tl_stop/valid",
                "tl_stop/state",
                "tl_stop/pos",
                "tl_stop/dir",
            ]
            wrapped_model = TracingWrapper(trainer.model, input_keys, self.batch_idx)

            # input_samples_list = list(self.input_sample.values())
            input_tuple = tuple(self.input_sample[key] for key in input_keys)
            # traced_model = torch.jit.trace(pl_module, input_samples_list)
            traced_model = torch.jit.trace(wrapped_model, input_tuple)

            # Save the traced model
            traced_model_path = Path(self.log_dir) / "traced_model.pt"
            traced_model.save(traced_model_path)

            # Log the model to wandb as an artifact
            artifact = wandb.Artifact(
                name=f"{logger.experiment.id}_traced_model", type="model"
            )
            artifact.add_file(str(traced_model_path), name="traced_model.pt")
            logger.experiment.log_artifact(artifact, aliases=["traced"])
            print(f"Traced model saved and uploaded to W&B: {traced_model_path}")
