# Softmax-5-layer implementation on my own
# date: July 13th 2017
# Author: Rui Wang

# import packages
import tensorflow as tf
# import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Read data
M_data = input_data.read_data_sets("/data/mnist", one_hot=True)

# define some parameters for the model
learning_rate = 0.003
batch_size = 100
n_epochs = 10
n_class = 10

# define the feature maps of each fully connected layer
A = 200
B = 100
C = 60
D = 30
# define the placeholders fo the model
X = tf.placeholder(tf.float32, [batch_size, 784])  # model input
Y_Practical = tf.placeholder(tf.float32, [batch_size, n_class])  # model practical labels

# define the weights and basis of the five layer model
# first connected layer
W1 = tf.Variable(tf.truncated_normal([784, A], stddev=0.1))
B1 = tf.Variable(tf.zeros([A]))

# second connected layer
W2 = tf.Variable(tf.truncated_normal([A, B], stddev=0.1))
B2 = tf.Variable(tf.zeros([B]))

# third connected layer
W3 = tf.Variable(tf.truncated_normal([B, C], stddev=0.1))
B3 = tf.Variable(tf.zeros([C]))

# fourth connected layer
W4 = tf.Variable(tf.truncated_normal([C, D], stddev=0.1))
B4 = tf.Variable(tf.zeros([D]))

# softmax layer
W5 = tf.Variable(tf.truncated_normal([D, n_class], stddev=0.1))
B5 = tf.Variable(tf.zeros([n_class]))

# concat all the weights and basis in order to activate all the independent input point
all_weights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)  # 0 = line concat
all_basis = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)

# define the model of each layer
Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
Y5 = tf.matmul(Y4, W5) + B5  # softmax layer

# compute the cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y5, labels=Y_Practical)
loss = tf.reduce_mean(cross_entropy)
# train the model
train_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Run on the Session
sum_each_batch = tf.Variable(0.0, tf.float32)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  # initialize the parameters
    n_batch = int(M_data.train.num_examples/batch_size)
    for i in range(n_epochs):
        for _ in range(n_batch):
            X_batch, Y_batch = M_data.train.next_batch(batch_size)
            w_value, b_value, op, loss_value = sess.run([all_weights, all_basis, train_optimizer, loss], feed_dict={X: X_batch, Y_Practical: Y_batch})
            sum_each_batch = sess.run(tf.add(sum_each_batch, loss_value))
        print("-------------iteration %d" % (i) + "  of loss value is %f" % (sum_each_batch/n_batch))

    # test model
    n_batch_test = int(M_data.test.num_examples / batch_size)
    correct_sample = 0
    for j in range(n_batch_test):
        X_batch_test, Y_batch_test = M_data.test.next_batch(batch_size)
        tr_opt_value, Y_output = sess.run([train_optimizer, Y5], feed_dict={X: X_batch_test, Y_Practical: Y_batch_test})
        sft_max_output = tf.nn.softmax(Y_output)
        predict_able = tf.equal(tf.argmax(sft_max_output, 1), tf.argmax(Y_batch_test, 1))
        predict_able_num = tf.reduce_sum(tf.cast(predict_able, tf.float32))
        correct_sample += sess.run(predict_able_num)
    print("The final classification accuracy of the 10 epochs is %f" % (correct_sample / M_data.test.num_examples))





