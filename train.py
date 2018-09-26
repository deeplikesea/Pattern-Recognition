import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os
'''
writer.write(example.SerializeToString()) input 最后生成数据。

从TFRecords文件中读取数据， 
首先需要用tf.train.string_input_producer生成一个解析队列。
之后调用tf.TFRecordReader的tf.parse_single_example解析器。如下图：

'''
IMAGE_SIZE = 128
NUM_CHANNELS = 3
batch_size=50
# CONV1_DEEP = 10
CONV1_DEEP = 10
CONV1_SIZE = 5
# CONV2_DEEP = 16
CONV2_DEEP = 16
CONV2_SIZE = 5
#FC1
FC_SIZE = 1024
#FC_OUTPUT
NUM_LABELS = 2

LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
num_train = 5001
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"#保存路径
MODEL_NAME = "model.ckpt"#用于save tf的session,我们需要使用后缀ckpt

def distort_color(image,color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32 / 255)
        image = tf.image.random_saturation(image, lower=0.5,upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower = 0.5,upper = 1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image,max_delta=32/255)
        image = tf.image.random_contrast(image, lower = 0.5,upper = 1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32 / 255)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image,0.0,1.0)

def preprocess_for_train(image):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(distorted_image,np.random.randint(2))#随机产生0-2之间的随机数
    return distorted_image

def read_and_decode(filename,batch_size):
    files = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(files, shuffle=True)
    '''
    将文件名列表交给tf.train.string_input_producer 
    函数.string_input_producer来生成一个先入先出的队列， 文件阅读器会需要它来读取数据。
    '''
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'channels': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })  
    '''
    取出包含image和label的feature对象
    因为之前用dictionary封装，所以提取数据简单很多。
    '''
    image, label = features['img_raw'], features['label']
    height, width = features['height'], features['width']
    channels = features['channels']

    # height = tf.cast(height, tf.uint8)
    # width = tf.cast(width, tf.uint8)
    decoded_image = tf.decode_raw(image, tf.uint8)#uint8是指0~2^8-1 = 255数据类型,一般在图像处理中很常见。 
    decoded_image = tf.reshape(decoded_image, [IMAGE_SIZE,IMAGE_SIZE,3])
    decoded_image = preprocess_for_train(decoded_image)
    min_after_dequeue = 100
    capacity = 1000 + 3 * batch_size
    '''
    capacity是队列的长度
    min_after_dequeue是出队后，队列至少剩下min_after_dequeue个数据
    '''
    image_batch, label_batch = tf.train.shuffle_batch([decoded_image, label], 
                                                      batch_size=batch_size, 
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    image_batch = tf.cast(image_batch, tf.float32)
    return  image_batch, label_batch
        
def inference(input_img):#网络结构
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        #conv1_size*conv1_size的矩阵内积相乘，从NUM_CHANNELS产生出CONV1_DEEP新的feature.
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))#正太分布初始，去除extreme值
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_img, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')#strides[1],[2]表示一格格移动。
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        #print(relu1.get_shape())
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#ksize定义2*2的max_pooling，stride依旧步长
        #print(pool1.get_shape())
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        #print(relu2.get_shape())
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #print(pool2.get_shape())
    pool_shape = pool2.get_shape().as_list()# 因为get_shape()返回的不是tensor 或string,而是元组,tf会出错，所以需要转list。  
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #print(nodes)
    reshaped = tf.reshape(pool2, [-1, nodes])#把之前的数据拉平方便FC，-1表示不管几个sample,这里是[n_sample,32*32*16]变成[n_sample,32*32*16]
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0001)(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0001)(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    return logit

def component(input_img):#最后一层输出图片
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        #conv1_size*conv1_size的矩阵内积相乘，从NUM_CHANNELS产生出CONV1_DEEP新的feature.
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))#正太分布初始，去除extreme值
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_img, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')#strides[1],[2]表示一格格移动。
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        #print(relu1.get_shape())
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#ksize定义2*2的max_pooling，stride依旧步长
        #print(pool1.get_shape())
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        #print(relu2.get_shape())
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #print(pool2.get_shape())
    return pool2
#%%
def train(img,label):
    logit = inference(img)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label)#把softmax和cross_entropy合起来了。
    #tf.nn.sigmoid_cross_entropy_with_logits -> this is for 2 class
    #tf.nn.softmax_cross_entropy_with_logits -> for multilables
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # #正则L2表达式
    # loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    loss = cross_entropy_mean
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)#平均滑动模型,num_updates = global_step
    variable_averages_op = variable_averages.apply(tf.trainable_variables())#应用平均滑动模型，增在varibale更新robust
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 3500 / batch_size, LEARNING_RATE_DECAY)

    #train_step = tf.train.FtrlOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # correct_prediction = tf.equal(tf.argmax(logit,1), label_batch)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:  # 开始一个会话
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(num_train):
            _,loss_Value, _, step = sess.run([train_op, loss, train_step, global_step])#每运行一次global_step加一
            print("----------------------")
            print("After %d training step(s),loss on training batch is %g." % (step, loss_Value))
            if i%1000 == 0:
                print("After %d training step(s),loss on training batch is %g." % (step, loss_Value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
            print("---------------------")

        coord.request_stop()
        coord.join(threads)
        
def main(argv=None):
    image_batch, label_batch = read_and_decode("train.tfrecords", batch_size=batch_size)
    train(image_batch,label_batch)
    # test_batch, testlabel_batch = read_and_decode("train2.tfrecords", batch_size=10*batch_size)
    # evaluate(test_batch,testlabel_batch)

if __name__ == '__main__':
    tf.app.run()

