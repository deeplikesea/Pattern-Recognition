import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
'''
tfrecord数据文件是一种将图像数据和标签统一存储的二进制文件，能更好的利用内存，在tensorflow中快速的复制，移动，读取，存储等。
tfrecord文件包含了tf.train.Example 协议缓冲区(protocol buffer，协议缓冲区包含了特征 Features)。你可以写一段代码获取你的数据,
将数据填入到Example协议缓冲区(protocol buffer)，将协议缓冲区序列化为一个字符串,
并且通过tf.python_io.TFRecordWriter class写入到TFRecords文件。
tensorflow/g3doc/how_tos/reading_data/convert_to_records.py就是这样的一个例子。

tf.train.Example中包含了属性名称到取值的字典，
其中属性名称为字符串，属性的取值可以为字符串（BytesList）、实数列表（
FloatList）或者整数列表（Int64List）。

example = tf.train.Example()这句将数据赋给了变量example（可以看到里面是通过字典结构实现的赋值）,
然后用writer.write(example.SerializeToString()) 这句实现写入。

将数据保存为tfrecord格式
首先需要给定tfrecord文件名称，并创建一个文件：
tfrecords_filename = './tfrecords/train.tfrecords'writer = tf.python_io.TFRecordWriter(tfrecords_filename) # 创建.tfrecord文件，准备写入
之后就可以创建一个循环来依次写入数据：
'''

'''
符合格式的file: train/ -> cat/ &dog/ -> picture
'''
IMAGE_SIZE = 128

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
'''
每一个example的feature成员变量是一个dict，存储一个样本的不同部分（例如图像像素+类标）。以下例子的样本中包含三个键a,b
'''

def get_files(filename):
    classes = []
    class_train = []
    label_train = []
    for train_class in os.listdir(filename):#读取文档里面的内容
        classes.append(train_class)
    for index,name in enumerate(classes):#0 for cat and 1 for dog
        '''
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        list(enumerate(seasons))
        [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
        '''
        for pic in os.listdir(filename+name):
            class_train.append(filename+name+'/'+pic)
            label_train.append(index)
    temp = np.array([class_train, label_train])#图像就是一个矩阵，在OpenCV for Python中，图像就是NumPy中的数组！
    temp = temp.transpose()  #shuffle the samp 
    np.random.shuffle(temp)  #after transpose,
    # images is in dimension 0 and label in dimension 1
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    writer = tf.python_io.TFRecordWriter("test.tfrecords")
    '''
    使用tf.python_io.TFRecordWriter创建一个专门存储tensorflow数据的writer，扩展名为’.tfrecord’。 
    该文件中依次存储着序列化的tf.train.Example类型的样本。
    '''
    for i in range(len(image_list)):
        img = cv2.imread(image_list[i])#读取bmp、jpg、png、tiff等常用格式。eg:img = cv2.imread("D:\cat.jpg")
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        height, width, channels = img.shape[0], img.shape[1], img.shape[2]
        img_raw = img.tostring()#返回一个使用标准“raw”编码器生成的包含像素数据的字符串。 string
        example = tf.train.Example(features=tf.train.Features(feature={#把数据做成字典形式
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'channels': _int64_feature(channels),
            'label': _int64_feature(int(label_list[i])),
            'img_raw': _bytes_feature(img_raw)}))
        writer.write(example.SerializeToString())
        #将数据封装成Example结构并写入TFRecord文件
    writer.close()
if __name__ == '__main__':
    get_files('./test/')