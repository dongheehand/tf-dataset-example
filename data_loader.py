import tensorflow as tf
import numpy as np
import argparse
import os

class data_loader():
    
    def __init__(self, conf):
        
        self.batch_size = args.batch_size
        self.in_memory = args.in_memory
        self.channel = args.channel
        self.tf_records = args.tf_records
        if self.in_memory:
            self.width = args.width
            self.height = args.height
            self.image_arr = tf.placeholder(shape = [None, self.height, self.width, self.channel], dtype = tf.uint8)
        
        if not self.in_memory or self.tf_records :
            self.image_arr  = np.array([os.path.join(args.data_direc, ele) for ele in sorted(os.listdir(args.data_direc))])
            
    def build_loader(self):
        
        if not self.tf_records:
            self.tr_dataset = tf.data.Dataset.from_tensor_slices(self.image_arr)
            
            if not self.in_memory:
                self.tr_dataset = self.tr_dataset.map(self._parse, num_parallel_calls = 4).prefetch(32)
        
        else:
            self.tr_dataset = tf.data.TFRecordDataset(self.image_arr)
            self.tr_dataset = self.tr_dataset.map(self._tf_record_parse, num_parallel_calls = 4).prefetch(32)
        
        self.tr_dataset = self.tr_dataset.shuffle(32)
        self.tr_dataset = self.tr_dataset.repeat()
        self.tr_dataset = self.tr_dataset.batch(self.batch_size)
        iterator = tf.data.Iterator.from_structure(self.tr_dataset.output_types, self.tr_dataset.output_shapes)
        self.next_batch = iterator.get_next()
        self.init_op = iterator.make_initializer(self.tr_dataset)

            
    def _parse(self, image):
        
        image = tf.read_file(image)
        image = tf.image.decode_png(image, channels = self.channel)
        
        return image
    
    def _tf_record_parse(self, example):
        
        feature = {'img' : tf.FixedLenFeature([], tf.string),
                  'height' : tf.FixedLenFeature([], tf.int64),
                  'width' : tf.FixedLenFeature([], tf.int64)}
        
        parsed_feature = tf.parse_single_example(example, feature)
        
        img = tf.decode_raw(parsed_feature['img'], tf.uint8)        
        height = tf.cast(parsed_feature['height'], tf.int32)
        width = tf.cast(parsed_feature['width'], tf.int32)
        
        img = tf.reshape(img, (height, width, self.channel))
        
        return img
    


if __name__ == '__main__':

    from PIL import Image

    def str2bool(v):
        return v.lower() in ('true')

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, default = 4)
    parser.add_argument("--in_memory", type = str2bool, default = False)
    parser.add_argument("--channel", type = int, default = 3)
    parser.add_argument("--width", type = int, default = 600)
    parser.add_argument("--height", type = int, default = 600)
    parser.add_argument("--data_direc", type = str, default = './test_data/')
    parser.add_argument("--result_path", type = str, default = './result/')
    parser.add_argument("--tf_records", type = str2bool, default = False)
    parser.add_argument("--test_num", type = int, default = 3)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    data_loader = data_loader(args)
    data_loader.build_loader()
    
    tf_output = data_loader.next_batch
    tf_output = tf_output[:,:50,:50,:]
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    
    if args.in_memory and not args.tf_records:
        img_list = []
        for ele in sorted(os.listdir(args.data_direc)):
            img_list.append(np.array(Image.open(os.path.join(args.data_direc, ele))))
        img_list = np.array(img_list)
        
        sess.run(data_loader.init_op, feed_dict = {data_loader.image_arr : img_list})
    else:
        sess.run(data_loader.init_op)
        
    count = 0
    
    for i in range(args.test_num):
        
        output = sess.run(tf_output)
        for ele in output:
            img = Image.fromarray(ele)
            img.save(os.path.join(args.result_path, '%02d_result.png'%count))
            count += 1

