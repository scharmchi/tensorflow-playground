import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# import input_data

x_t_input_vec_size = 28
time_step_size = 28  # this probably should be 1 if you want to generate next word in a sequence based on previous words
# here after seeing all (i.e. 28) rows/cols (each one has dim 28) of the input image we decide to generate output
n_hidden = 128
n_classes = 10

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# I guess time_step_size is like rows of the image and
# input vec is each row which has the size equal to number of columns
x = tf.placeholder("float", [None, time_step_size, x_t_input_vec_size])
y = tf.placeholder("float", [None, n_classes])
init_state = tf.placeholder("float", [None, 2 * n_hidden])  # state and cell? 2 * n_hidden?

W_hidden = tf.Variable(tf.random_normal([x_t_input_vec_size, n_hidden], stddev=0.01))
b_hidden = tf.Variable(tf.random_normal([n_hidden], stddev=0.01))

W_out = tf.Variable(tf.random_normal([n_hidden, n_classes], stddev=0.01))
b_out = tf.Variable(tf.random_normal([n_classes], stddev=0.01))

# right now the shape is [batch_size, num_time_steps, num_inputs]
# so now we permute num_time_steps and batch_size so we get the shape
# [num_time_steps, batch_size, num_inputs]
x_censored = tf.transpose(x, [1, 0, 2])

# and now prepare it for input to hidden so that it becomes of shape
# [num_time_steps * batch_size, num_inputs]
x_censored = tf.reshape(x_censored, [-1, x_t_input_vec_size])

x_censored = tf.matmul(x_censored, W_hidden) + b_hidden

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

# Split data because rnn cell needs a list of inputs for the RNN inner loop
x_censored = tf.split(0, time_step_size, x_censored)

outputs, states = tf.nn.rnn(lstm_cell, x_censored, initial_state=init_state)
pred_vec = tf.matmul(outputs[-1], W_out) + b_out

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_vec, y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred_vec, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

batch_size = 128
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    step = 1
    while step * batch_size < 100000:
        train_batch_x, train_batch_y = mnist.train.next_batch(128)
        # reshape from 784 -> 28x28 | each row is a step and consists of a 28-elem vector as the input
        train_batch_x = train_batch_x.reshape((batch_size, time_step_size, x_t_input_vec_size))
        # print(train_batch_y.__class__)
        sess.run(optimizer, feed_dict={x: train_batch_x, y: train_batch_y,
                                       init_state: np.zeros((batch_size, 2*n_hidden))})
        if step % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: train_batch_x, y: train_batch_y,
                                                init_state: np.zeros((batch_size, 2*n_hidden))})
            loss = sess.run(cost, feed_dict={x: train_batch_x, y: train_batch_y,
                                             init_state: np.zeros((batch_size, 2*n_hidden))})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) +
                  ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Done!!")
    num_test = 256
    test_x = mnist.test.images[:num_test].reshape((-1, time_step_size, x_t_input_vec_size))
    test_y = mnist.test.labels[:num_test]
    print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x:test_x, y: test_y,
                                                              init_state: np.zeros((num_test, 2*n_hidden))}))

