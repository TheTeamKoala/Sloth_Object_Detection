import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import schedule
import time

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util


MODEL_NAME = 'inference_graph(2)'

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

NUM_CLASSES = 15

all_objects = []


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def job():
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    camera = cv2.VideoCapture(0)

    ret = camera.set(3, 1280)
    ret = camera.set(4, 720)

    objects = []
    print("I'm working...")
    for i in range(5):
        ret, frame = camera.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        for index, value in enumerate(classes[0]):
            if scores[0, index] > 0.75:
                if not objects.__contains__((category_index.get(value)).get('name')):
                    objects.append((category_index.get(value)).get('name'))
                if not all_objects.__contains__((category_index.get(value)).get('name')):
                    all_objects.append((category_index.get(value)).get('name'))

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=7,
            min_score_thresh=0.75)

        # cv2.imshow('Object detector', frame)
        if len(objects) > 0:
            cv2.imwrite('image.png', frame)

    diff_list = diff(all_objects, objects)
    for obj in objects:
        print(obj + " added.")
        os.system('python add.py ' + obj)

    for diff_object in diff_list:
        print(diff_object + " deleted.")
        os.system('python mainPy.py delete ' + diff_object)
        all_objects.remove(diff_object)

    camera.release()
    cv2.destroyAllWindows()


schedule.every(1).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
