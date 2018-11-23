import tensorflow as tf
from PIL import Image
import numpy as np
import argparse
import os


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

if __name__ == '__main__':
    
    def str2bool(v):
        return v.lower() in ('true')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_direc", type = str, default = './test_data/')
    parser.add_argument("--record_path", type = str, default = './tf_records')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.record_path):
        os.mkdir(args.record_path)
    
    img_list = sorted(os.listdir(args.data_direc))
    
    for i, ele in enumerate(img_list):
        img_ = np.array(Image.open(os.path.join(args.data_direc, ele)))
        h, w, _ = img_.shape
        img_ = img_.tostring()
        writer = tf.python_io.TFRecordWriter(os.path.join(args.record_path, 'img_%04d.tfrecords' % i))
        img_ = tf.train.Example(features = tf.train.Features(feature = {'img' : _bytes_feature(img_),
                                                                       'height' : _int64_feature(h),
                                                                       'width' : _int64_feature(w)}))
        writer.write(img_.SerializeToString())
        writer.close()

    

