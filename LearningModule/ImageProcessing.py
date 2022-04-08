import numpy
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from LearningModule.DetectObject import *

WORKSPACE_PATH = '/home/kansh_dev/PycharmProjects/Signius/LearningModule/workspace'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'


class ImageProcessing:
    """restoring dictionary of gestures into category_index"""
    category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt',
                                                                        use_display_name=True)
    def ret_processed_image(self, frame: numpy.ndarray) -> tuple:

        """processing frame into tensor"""
        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        """detection dictionary contains each prediction, for each of them we have gesture id,
         obtained score and 4 points describing the box where the gesture is located in the image"""
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

        """minor data transformations"""
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        processed_image = image_np.copy()

        """Because in detections dict we have a lot of predictions, we are getting the most accurate one, and saving it,
         when propability is higher than 50%"""
        predicted_gestures_id = detections['detection_classes'] + 1
        max_thresh_score = list(detections['detection_scores'])[0]
        most_acc_gesture = 0
        if max_thresh_score >= 0.5:
            most_acc_gesture = list(predicted_gestures_id)[0]

        """Here we basically modify the image by applying a frame and a prediction to it"""
        viz_utils.visualize_boxes_and_labels_on_image_array(
            processed_image,
            detections['detection_boxes'],
            detections['detection_classes'] + 1,
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=1,  # Here we can change number of predictions on one image
            min_score_thresh=.5,  # Here we change the minimum probability to draw prediction on image
            agnostic_mode=False)

        return processed_image, most_acc_gesture
