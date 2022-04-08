import tensorflow as tf
import os
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# paths to necessary files
WORKSPACE_PATH = '/home/kansh_dev/PycharmProjects/Signius/LearningModule/workspace'
APIMODEL_PATH = 'models'
MODEL_PATH = WORKSPACE_PATH + '/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
MODEL_NAME = 'my_ssd_mobnet'
CONFIG_PATH = MODEL_PATH + '/' + MODEL_NAME + '/pipeline.config'

# modifying pipeline.config file with our specifications, paths etc. to prepare model to train
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, 'r') as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = 10  # number of classes
pipeline_config.train_config.batch_size = 5  # determining the quality of the model depending on the computational power

# paths to some folders/data to configure (names say exactly which path should be putted, here's absolutely
# everything needed to work, bot of course we can change more parameters, experiment with our model to get better
# results, for example batch size which is not necessary to configure
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH + \
                                                    '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
    f.write(config_text)

# when pipeline_config is saved with our parameters, we can basically run function to train model, the important thing
# is '--num_train_steps' which tell us how many steps our model should do, at the begining it's good to start with something
# like 5k steps, in some cases it's enough to prepare pretty good model

# running model_main_tf2.py file from TensorFlow to create model which recognise our gestures
os.system("""python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=30000""".
          format(APIMODEL_PATH, MODEL_PATH, MODEL_NAME, MODEL_PATH, MODEL_NAME))
