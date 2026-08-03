from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from .config import GPU_DEVICE_ID  # noqa: F401 - configures CUDA before TensorFlow import
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


@tf.keras.utils.register_keras_serializable(package="Custom")
class TakeLastTimestep(layers.Layer):
    def call(self, x):
        return x[:, -1]

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="Custom")
class TileToHorizon(layers.Layer):
    def __init__(self, horizon=7, **kwargs):
        super().__init__(**kwargs)
        self.horizon = int(horizon)

    def call(self, x):
        return tf.tile(tf.expand_dims(x, axis=1), [1, self.horizon, 1, 1, 1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"horizon": self.horizon})
        return cfg


CUSTOM_OBJECTS = {
    "TakeLastTimestep": TakeLastTimestep,
    "Custom>TakeLastTimestep": TakeLastTimestep,
    "TileToHorizon": TileToHorizon,
    "Custom>TileToHorizon": TileToHorizon,
}


@dataclass
class LoadedModelBundle:
    sm_model: keras.Model
    et_model: keras.Model
    sm_norms: Dict[str, Any]
    et_norms: Dict[str, Any]
    manifest: Dict[str, Any]
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def resolve_timescale_model_dir(base_dir: str, parent_dirs: list[str], best_arch_folder: str) -> str:
    for p in parent_dirs:
        cand = os.path.join(base_dir, p, best_arch_folder)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        f"Could not find model folder. Tried: "
        + ", ".join(os.path.join(base_dir, p, best_arch_folder) for p in parent_dirs)
    )


def _load_norms(path: str) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(model_dir: str) -> Dict[str, Any]:
    path = os.path.join(model_dir, "model_manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing selected-model manifest: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != "seed_target_specific_model_bundle/v1":
        raise ValueError(f"Unsupported model manifest schema in {path}")
    if manifest.get("input_variables") != ["PRECTmms", "TBOT", "WIND", "QBOT", "PSRF", "FSDS", "FLDS"]:
        raise ValueError(f"Model manifest does not declare the seven required NLDAS variables: {path}")
    return manifest


def _verify_manifest_files(model_dir: str, manifest: Dict[str, Any]) -> None:
    for target in ("ET", "SM"):
        target_meta = manifest.get("targets", {}).get(target)
        if not isinstance(target_meta, dict) or target_meta.get("input_channels") != 7:
            raise ValueError(f"Invalid {target} model metadata in {model_dir}")
        for kind in ("checkpoint", "normalizer"):
            artifact = target_meta.get(kind)
            if not isinstance(artifact, dict):
                raise ValueError(f"Missing {target} {kind} metadata in {model_dir}")
            path = os.path.join(model_dir, str(artifact.get("filename")))
            expected = str(artifact.get("sha256"))
            if _sha256_file(path) != expected:
                raise ValueError(f"Selected-model artifact hash mismatch: {path}")


def _validate_loaded_contract(model: keras.Model, norms: Dict[str, Any], target: str, manifest: Dict[str, Any]) -> None:
    target_meta = manifest["targets"][target]
    shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    if len(shape) != 5 or int(shape[1]) != int(target_meta["input_days"]) or int(shape[-1]) != 7:
        raise ValueError(f"{target} model input shape violates selected contract: {shape}")
    output_shape = model.output_shape[0] if isinstance(model.output_shape, list) else model.output_shape
    if len(output_shape) not in (4, 5) or int(output_shape[-1]) != 1:
        raise ValueError(f"{target} model output shape is not an endpoint field: {output_shape}")
    for key in ("input_mu", "input_sd", "target_mu", "target_sd"):
        if key not in norms or not np.all(np.isfinite(norms[key])):
            raise ValueError(f"{target} normalizer has missing or non-finite {key}")
    if np.asarray(norms["input_mu"]).reshape(-1).size != 7 or np.asarray(norms["input_sd"]).reshape(-1).size != 7:
        raise ValueError(f"{target} normalizer does not contain exactly seven input channels")


def load_models(model_dir: str) -> LoadedModelBundle:
    sm_model_path = os.path.join(model_dir, "keras_convlstm_sm_best.keras")
    et_model_path = os.path.join(model_dir, "keras_convlstm_et_best.keras")
    sm_norms_path = os.path.join(model_dir, "keras_convlstm_sm_norms.npz")
    et_norms_path = os.path.join(model_dir, "keras_convlstm_et_norms.npz")

    manifest = _load_manifest(model_dir)
    for p in [sm_model_path, et_model_path, sm_norms_path, et_norms_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")
    _verify_manifest_files(model_dir, manifest)

    sm_model = keras.models.load_model(sm_model_path, compile=False, custom_objects=CUSTOM_OBJECTS)
    et_model = keras.models.load_model(et_model_path, compile=False, custom_objects=CUSTOM_OBJECTS)

    sm_norms = _load_norms(sm_norms_path)
    et_norms = _load_norms(et_norms_path)
    _validate_loaded_contract(sm_model, sm_norms, "SM", manifest)
    _validate_loaded_contract(et_model, et_norms, "ET", manifest)

    return LoadedModelBundle(
        sm_model=sm_model,
        et_model=et_model,
        sm_norms=sm_norms,
        et_norms=et_norms,
        manifest=manifest,
    )
