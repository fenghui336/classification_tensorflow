import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt
cwd = '/Users/fenghui/PycharmProjects/vgg_tf/image/train/'
classes = {'apple','orange'}


def createdata():
    filename="./data/myself_train.tfrecords"      #要生成的文件名以及地址，不指定绝对地址的话就是在建立在工程目录下
    writer = tf.python_io.TFRecordWriter(filename)  # 使用该函数创建一个tfrecord文件
    height = 256
    width = 256
    for index, name in enumerate(classes):
        class_path = cwd + name + '/'
        for img_name in os.listdir(class_path): # 以list的方式显示目录下的各个文件夹
            img_path = class_path + img_name  # 每一个图片的地址
            img = Image.open(img_path)
            img = img.resize((height, width))
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()
def read_and_decode(filename, batch_size):  # 读取tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,  #对序列化数据进行解析
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])  # reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255)  # 将tensor数据转化为float32格式
    label = tf.cast(features['label'], tf.int32)  # 将label标签转化为int32格式
    label = tf.one_hot(label, 2)   #对标签做one hot处理：假如共有4个类，若标签为3，做one hot之后则为[0 0 0 1],若标签为0，则[1 0 0 0]
    # img_batch, label_batch = tf.train.batch([img,label],batch_size,1,50)
    img_batch, label_batch = tf.train.shuffle_batch([img,label],batch_size,500,100)     #打乱排序输出batch
    return img_batch, label_batch


if __name__ == "__main__":
    createdata()
    '''
    init_op = tf.global_variables_initializer()
    image, label = read_and_decode("myself_train.tfrecords", 32)
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(10):
            example, l = sess.run([image, label])  # 取出image和label
            plt.imshow(example[i, :, :, :])
            plt.show()
            print('label:',l)
            print(example.shape)
        coord.request_stop()
        coord.join(threads)

'''
