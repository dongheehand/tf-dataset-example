
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import argparse
import os


# In[ ]:


class data_loader():
    
    def __init__(self, conf):
        
        self.batch_size = conf['batch_size']
        self.in_memory = conf['in_memory']
        self.channel = conf['channel']
        
        if self.in_memory:
            self.width = conf['width']
            self.height = conf['height']
            self.image_arr = tf.placeholder(shape = [None, self.height, self.width, self.channel], dtype = tf.uint8)
        
        if not self.in_memory:
            self.image_arr  = conf['data_path']
            
    def build_loader(self):

        self.tr_dataset = tf.data.Dataset.from_tensor_slices(self.image_arr)

        if not self.in_memory :
            self.tr_dataset = self.tr_dataset.map(self._parse, num_parallel_calls = 4).prefetch(32)

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


# In[ ]:


from PIL import Image

if __name__ == '__main__':
    
    def str2bool(v):
        return v.lower() in ('true')

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, default = 4)
    parser.add_argument("--in_memory", type = str2bool, default = False)
    parser.add_argument("--channel", type = int, default = 3)
    parser.add_argument("--width", type = int, default = 64)
    parser.add_argument("--height", type = int, default = 64)
    parser.add_argument("--data_direc", type = str, default = './test_data/')
    parser.add_argument("--result_path", type = str, default = './result')
    parser.add_argument("--test_num", type = int, default = 3)
    
    args = parser.parse_args()

    conf = {}
    
    conf['batch_size'] = args.batch_size
    conf['in_memory'] = args.in_memory
    conf['channel'] = args.channel
    conf['width'] = args.width
    conf['height'] = args.height
    conf['data_direc'] = args.data_direc
    conf['result_path'] = args.result_path
    conf['test_num'] = args.test_num
    
    img_list = sorted(os.listdir(conf['data_direc']))
    conf['data_path'] = np.array([os.path.join(conf['data_direc'], ele) for ele in img_list])
    
    if not os.path.exists(conf['result_path']):
        os.makedirs(conf['result_path'])
    
    data_loader = data_loader(conf)
    data_loader.build_loader()
    
    tf_output = data_loader.next_batch
    tf_output = tf_output[:,:50,:50,:]
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    
    if conf['in_memory']:
        img_list = []
        for ele in conf['data_path']:
            img_list.append(np.array(Image.open(ele)))
        img_list = np.array(img_list)
        
        sess.run(data_loader.init_op, feed_dict = {data_loader.image_arr : img_list})
    else:
        sess.run(data_loader.init_op)
        
    count = 0
    
    for i in range(conf['test_num']):
        
        output = sess.run(tf_output)
        for ele in output:
            img = Image.fromarray(ele)
            img.save(os.path.join(conf['result_path'], '%02d_result.png'%count))
            count += 1

