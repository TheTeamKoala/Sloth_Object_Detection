"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'fanta':
        return 1
    elif row_label == 'gazoz':
        return 2
    elif row_label == 'hardal':
        return 3
    elif row_label == 'kasar':
        return 4
    elif row_label == 'ketcap':
        return 5
    elif row_label == 'kola':
        return 6
    elif row_label == 'limonata':
        return 7
    elif row_label == 'mayonez':
        return 8
    elif row_label == 'meyvesuyu':
        return 9
    elif row_label == 'peynir':
        return 10
    elif row_label == 'soda':
        return 11
    elif row_label == 'sut':
        return 12
    elif row_label == 'tereyag':
        return 13
    elif row_label == 'yogurt':
        return 14
    elif row_label == 'yumurta':
        return 15
    elif row_label == 'elma':
        return 16
    elif row_label == 'kivi':
        return 17
    elif row_label == 'karpuz':
        return 18
    elif row_label == 'marul':
        return 19
    elif row_label == 'turp':
        return 20
    elif row_label == 'patlican':
        return 21
    elif row_label == 'biber':
        return 22
    elif row_label == 'salatalik':
        return 23
    elif row_label == 'limon':
        return 24
    elif row_label == 'portakal':
        return 25
    elif row_label == 'lahana':
        return 26
    elif row_label == 'muz':
        return 27
    elif row_label == 'cilek':
        return 28
    elif row_label == 'havuc':
        return 29
    elif row_label == 'domates':
        return 30
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    img_dirs = ['fanta', 'fanta1', 'gazoz', 'gazoz1', 'hardal', 'kasar', 'ketcap', 'kola', 'limonata', 'mayonez',
                'meyvesebze', 'meyvesuyu', 'peynir', 'soda', 'sut', 'tereyag', 'yogurt', 'yumurta', 'zzz']
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    for i in img_dirs:
        path = os.path.join(os.getcwd(), (FLAGS.image_dir + '/' + i))
        examples = pd.read_csv(FLAGS.csv_input + '/' + i + '_labels.csv')
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
