from __future__ import annotations

from .config import GPU_DEVICE_ID  # noqa: F401 - configures CUDA before TensorFlow import
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Custom", name="TakeLastTimestep")
class TakeLastTimestep(tf.keras.layers.Layer):
    """selects the final timestep from [batch, time, height, width, channels]"""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs[:, -1, ...]

    def get_config(self):
        return super().get_config()


def get_custom_objects() -> dict[str, object]:
    return {"TakeLastTimestep": TakeLastTimestep}
