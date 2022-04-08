import os

"""paths to necessary files"""
WORKSPACE_PATH = '/home/kansh_dev/PycharmProjects/Signius/LearningModule/workspace'
SCRIPTS_PATH = 'scripts'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'

"""labels for each sign we want to recognise"""
labels = [{'name': 'aGesture', 'id': 1},
          {'name': 'bGesture', 'id': 2},
          {'name': 'cGesture', 'id': 3},
          {'name': 'dGesture', 'id': 4},
          {'name': 'eGesture', 'id': 5},
          {'name': 'fGesture', 'id': 6},
          {'name': 'gGesture', 'id': 7},
          {'name': 'hGesture', 'id': 8},
          {'name': 'iGesture', 'id': 9},
          {'name': 'undoGesture', 'id': 10}]

"""creating labels map file, which contains all labels, which we defined previously"""
with open('workspace/annotations' + '/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item{\n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

"""generating tf records with paths to our photos using generate_tfrecord.py from TensorFlow"""
os.system(f"python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l"
          f" {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/train.record'}")
os.system(f"python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/test'} -l"
          f" {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/test.record'}")
