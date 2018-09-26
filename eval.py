import tensorflow as tf
import numpy as np
import os
import train
import time
import cv2
MOVING_AVERAGE_DECAY = 0.99
EVAL_INTERVAL_SECS = 10
image_size = 128
batch_size = 500

files = tf.train.match_filenames_once("train.tfrecords")
filename_queue = tf.train.string_input_producer(files, shuffle=True)
    # filename_queue = tf.train.string_input_producer(["dog_train.tfrecords0"]) #读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'channels': tf.FixedLenFeature([], tf.int64),
                                       })  # 取出包含image和label的feature对象
image, label = features['img_raw'], features['label']
height, width = features['height'], features['width']
height = tf.cast(height, tf.uint8)
width = tf.cast(width, tf.float32)
channels = features['channels']
decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image = tf.reshape(decoded_image, [image_size, image_size, 3])
    # image_size = 130

min_after_dequeue = 100
capacity = 1000 + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch([decoded_image, label], batch_size=batch_size,
                                                      capacity=capacity, min_after_dequeue=min_after_dequeue)

image_batch = tf.cast(image_batch, tf.float32)

y = train.inference(image_batch)
correct_prediction = tf.equal(tf.argmax(y,1),label_batch)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
variable_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variable_to_restore)
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
    print(ckpt)
    print(ckpt.model_checkpoint_path)
    print(ckpt.all_model_checkpoint_paths)
    for path in ckpt.all_model_checkpoint_paths:
        saver.restore(sess,path)
        global_step = int(path.split('/')[-1].split('-')[-1])-1
        accuracy_score = sess.run(accuracy)
        print("After %s training step(s),test accuracy = %g" % (global_step, accuracy_score))
    '''
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])-1
        for i in range(1):
            accuracy_score = sess.run(accuracy)
            print("After %s training step(s),test accuracy = %g" % (global_step, accuracy_score))
    else:
        print("No checkpoint file found")
    '''
    coord.request_stop()
    coord.join(threads)
    # time.sleep(EVAL_INTERVAL_SECS)


