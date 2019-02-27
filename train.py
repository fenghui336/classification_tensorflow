import tensorflow as tf

classes = 2
image_size = 256
train_path = './my_train.tfrecords'
#test_path = './myself_test.tfrecords'


def compute_accuracy(v_xs,v_ys):
    global prediction
    pred_y = sess.run(prediction,feed_dict={x:v_xs,keep_prob:1.})
    correct_accuracy = tf.equal(tf.argmax(pred_y,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_accuracy,tf.float32))
    result = sess.run(accuracy,feed_dict={x:v_xs,y:v_ys,keep_prob:1.})
    return result

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label':tf.FixedLenFeature([],tf.int64),
            'img_raw':tf.FixedLenFeature([],tf.string),
        }
    )
    img = tf.decode_raw(features['img_raw'],tf.uint8)
    img = tf.reshape(img,[image_size,image_size,3])
    img = tf.cast(img,tf.float32)*(1./255)
    label = tf.cast(features['label'],tf.int64)
    label = tf.one_hot(label,classes,dtype = tf.int64)
    img_batch,label_batch = tf.train.shuffle_batch([img,label],32,500,100,num_threads=4)
    return img_batch,label_batch



img_train,labels_train = read_and_decode(train_path)
img_test,labels_test = read_and_decode(train_path)


def weight_varibale(shape,f_name):
    initial = tf.truncated_normal(shape,mean=0,stddev=0.1)
    return tf.Variable(initial,name=f_name)
def bias_variable(shape,f_name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=f_name)
def conv2d_filter(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pooling_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32,[None,image_size,image_size,3],name='images')
y = tf.placeholder(tf.float32,[None,classes])
keep_prob = tf.placeholder(tf.float32)

#第一层卷积池化
W_conv1 = weight_varibale([5,5,3,32],'W_conv1')
b_conv1 = bias_variable([32],'b_conv1')
h_conv1 = tf.nn.relu(conv2d_filter(x,W_conv1)+b_conv1)
h_pool1 = max_pooling_2x2(h_conv1)
#第二层卷积池化
W_conv2 = weight_varibale([5,5,32,64],'W_conv2')
b_conv2 = bias_variable([64],'b_conv2')
h_conv2 = tf.nn.relu(conv2d_filter(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pooling_2x2(h_conv2)  #64*64*64
#第一层全连接
W_fc1 = weight_varibale([64*64*64,1024],'W_fc1')
b_fc1 = bias_variable([1024],'b_fc1')
h_pool2_flat = tf.reshape(h_pool2,[-1,64*64*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#第二层全连接
h_fc1_frop = tf.nn.dropout(h_fc1,keep_prob)
W_fc2 = weight_varibale([1024,classes],'W_fc2')
b_fc2 = bias_variable([classes],'b_fc2')
prediction = tf.nn.softmax(tf.matmul(h_fc1_frop,W_fc2)+b_fc2,name='prediction')


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=prediction,labels=y
),name='loss')
train = tf.train.AdamOptimizer(1e-4).minimize(loss,name='train')

with tf.Session() as sess:
    sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    for i in range(50):
        img_xs,label_xs = sess.run([img_train,labels_train])
        sess.run(train,feed_dict={
            x:img_xs,y:label_xs,keep_prob:0.7
        })

        if i%1 == 0:
            img_test_xs,label_test_xs = sess.run([img_test,labels_test])
            print(compute_accuracy(img_test_xs,label_test_xs))
    coord.request_stop()
    coord.join(threads)
    sess.close()
