#!/usr/bin/env python3
import dataclasses
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"[inspect_batch] Running on: {platform.node()}")
    logging.info(f"[inspect_batch] Config name: {config.name}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    # ----- 创建 dataloader 并取一个 batch -----
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)  # batch = (Observation, Actions)

    logging.info("[inspect_batch] Initialized data loader.")
    logging.info("[inspect_batch] ==== array_tree_to_info(batch) ====")
    logging.info("\n" + training_utils.array_tree_to_info(batch))

    obs, acts = batch

    # ----- 1) 观察 state -----
    state_arr = np.array(obs.state)
    logging.info(
        f"[inspect_batch] state shape={state_arr.shape}, "
        f"min={state_arr.min():.4f}, max={state_arr.max():.4f}, "
        f"mean={state_arr.mean():.4f}, std={state_arr.std():.4f}"
    )

    # ----- 2) 观察 actions -----
    def summarize(x):
        x_np = np.array(x)
        return dict(
            shape=x_np.shape,
            min=float(x_np.min()),
            max=float(x_np.max()),
            mean=float(x_np.mean()),
            std=float(x_np.std()),
        )

    logging.info("[inspect_batch] ==== actions pytree summary ====")
    logging.info(jax.tree.map(summarize, acts))

    # acts 本身就是 actions 数组
    a_main = np.array(acts)
    logging.info(f"[inspect_batch] main actions array shape = {a_main.shape}")
    logging.info(f"[inspect_batch] main actions dtype = {a_main.dtype}")

    if a_main.ndim >= 3:
        # Shape 通常是 (batch, seq_len, action_dim)
        logging.info(f"[inspect_batch] example actions[0, 0, :] = {a_main[0, 0, :]}")
        if a_main.shape[1] > 1:
            logging.info(f"[inspect_batch] example actions[0, 1, :] = {a_main[0, 1, :]}")
    elif a_main.ndim == 2:
        # Shape 是 (batch, action_dim)
        logging.info(f"[inspect_batch] example actions[0, :] = {a_main[0, :]}")
    else:
        logging.info(f"[inspect_batch] actions have shape: {a_main.shape}")

    logging.info("[inspect_batch] Done.")


if __name__ == "__main__":
    # 和训练脚本一样，从命令行拿 config_name，例如：
    # uv run scripts/inspect_batch.py --config_name pi05_npm_lora
    main(_config.cli())
