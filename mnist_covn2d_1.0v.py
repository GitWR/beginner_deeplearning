# Function: implementation of 1.0 Version CNN of Mnist Dataset
# Author: Rui Wang
# Date: 2017/08/01
# encoding: UTF-8
# import packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
# read in data
m_data = input_data.read_data_sets("/data/mnist", one_hot=True)
batch_size = 100  # how many a group
# define the placeholders
X = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])
Y_practical = tf.placeholder(tf.float32, [batch_size, 10])
lr = tf.placeholder(tf.float32)
p_drop = tf.placeholder(tf.float32)

n_class = 10  # categories num
epoch = 2  # iterations

# define numbers of feature maps of each layer
A = 4
B = 8
C = 12
D = 200

# Define the W and B, totally five layers, 3 cov layers, 1 fc layer, 1 soft_max layer
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, A], stddev=0.1))
B1 = tf.Variable(tf.ones([A])/n_class)

W2 = tf.Variable(tf.truncated_normal([5, 5, A, B], stddev=0.1))
B2 = tf.Variable(tf.ones([B])/n_class)

W3 = tf.Variable(tf.truncated_normal([4, 4, B, C], stddev=0.1))
B3 = tf.Variable(tf.ones([C])/n_class)

W4 = tf.Variable(tf.truncated_normal([2*2*C, D], stddev=0.1))
B4 = tf.Variable(tf.ones([D])/n_class)

W5 = tf.Variable(tf.truncated_normal([D, n_class], stddev=0.1))
B5 = tf.Variable(tf.ones([n_class])/n_class)

# build the simple CNN model
stride_1 = 1
cov1 = tf.nn.conv2d(X, W1, strides=[1, stride_1, stride_1, 1], padding='SAME')
Y1 = tf.nn.relu(cov1+B1)
pool_1 = tf.nn.max_pool(Y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 14 * 14
# Y1_drop = tf.nn.dropout(Y1, p_drop)

stride_2 = 2
cov2 = tf.nn.conv2d(pool_1, W2, strides=[1, stride_2, stride_2, 1], padding='SAME')
Y2 = tf.nn.relu(cov2+B2)
pool_2 = tf.nn.max_pool(Y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 4 * 4
Y2_drop = tf.nn.dropout(pool_2, p_drop)

stride_3 = 2
Y3 = tf.nn.relu(tf.nn.conv2d(Y2_drop, W3, strides=[1, stride_3, stride_3, 1], padding='SAME')+B3)
Y3_drop = tf.nn.dropout(Y3, p_drop)
Y3_reshape = tf.reshape(Y3_drop, shape=[-1, 2*2*C])

Y4 = tf.nn.relu(tf.matmul(Y3_reshape, W4)+B4)
Y4_drop = tf.nn.dropout(Y4, p_drop)

Y5 = tf.matmul(Y4_drop, W5)+B5
# Y5_sm = tf.nn.softmax(Y5)  # it will be used in the testing phrase

# train this simple CNN model
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y5, labels=Y_practical)
loss = tf.reduce_mean(cross_entropy)
train_optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

#  run on the created session()
all_weights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
all_basis = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
sum_each_batch = tf.Variable(0.0, tf.float32)  # use to compute the loss of each batch
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    n_batch = int(m_data.train.num_examples/batch_size)  # this dataset can be divided into how many groups
    for i in range(epoch):
        lr_min_value = 0.0003
        lr_max_value = 0.001
        decay_speed = 2000
        learning_rate = lr_min_value + (lr_max_value-lr_min_value)*math.exp(-i/decay_speed)
        for j in range(n_batch):
            X_batch, Y_batch = m_data.train.next_batch(batch_size)
            X_batch = np.reshape(X_batch, [batch_size, 28, 28, 1])
            W_value, B_value, loss_value = sess.run([all_weights, all_basis, loss], feed_dict={X: X_batch, Y_practical: Y_batch, p_drop: 0.5})
            opt_value = sess.run(train_optimizer, feed_dict={X: X_batch, Y_practical: Y_batch, lr: learning_rate, p_drop: 0.5})
            sum_each_batch = sess.run(tf.add(sum_each_batch, loss_value))
        print("-------------iteration : %d" % i + " of loss value is %f" % (sum_each_batch/n_batch))
        print("\n-------------learning rate step:%d" % i + "value is", learning_rate)

    # test model
    n_batch_test = int(m_data.test.num_examples/batch_size)
    correct_num = 0
    for k in range(n_batch_test):
        X_batch_test, Y_batch_test = m_data.test.next_batch(batch_size)
        X_batch_test = np.reshape(X_batch_test, [batch_size, 28, 28, 1])
        Y_output = sess.run(Y5, feed_dict={X: X_batch_test, Y_practical: Y_batch_test, p_drop: 0})
        Y5_sm = tf.nn.softmax(Y_output)
        predict_able = tf.equal(tf.argmax(Y5_sm, 1), tf.argmax(Y_batch_test, 1))
        predict_num = tf.reduce_sum(tf.cast(predict_able, tf.float32))
        correct_num += sess.run(predict_num)
    print("The final classification accuracy of the 100 epochs is %f" % (correct_num / m_data.test.num_examples))





