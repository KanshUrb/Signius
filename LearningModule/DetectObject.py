import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

WORKSPACE_PATH = '/home/kansh_dev/PycharmProjects/Signius/LearningModule/workspace'
MODEL_PATH = WORKSPACE_PATH + '/models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'

"""highest checkpoint identifier (last one is the most accurate)"""
ckpt_id = 'ckpt-31'

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

"""Restore checkpoint"""
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, ckpt_id)).expect_partial()


@tf.function
def detect_fn(image) -> dict:
    """function that converts a given image into a dictionary of predictions using our model and restoring the latest checkpoint (ckpt_id) which we got while we was creating model"""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
