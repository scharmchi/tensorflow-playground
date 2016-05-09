from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# strides = [batch, height, width, channels]
# strides means how much the window shifts by in each of the dimensions
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_kxk(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_kxk(h_conv1, 2)

# The first two dimensions are the patch size,
# the next is the number of input channels, and the last is the number of output channels
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_kxk(h_conv2, 2)

W_fc_1 = weight_variable([7 * 7 * 64, 1024])
b_fc_1 = bias_variable([1024])

input_to_fully = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc_1 = tf.nn.relu(tf.matmul(input_to_fully, W_fc_1) + b_fc_1)

dropout_keep_prob = tf.placeholder(tf.float32)
h_fc_1_drop = tf.nn.dropout(h_fc_1, dropout_keep_prob)

W_fc_2 = weight_variable([1024, 10])
b_fc_2 = bias_variable([10])

y_pred = tf.nn.softmax(tf.matmul(h_fc_1_drop, W_fc_2) + b_fc_2)

cross_entropy_cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_cost)

correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y: batch[1], dropout_keep_prob: 0.5})
    if i % 100 == 0:
        train_acc = accuracy.eval(feed_dict={x: batch[0], y: batch[1], dropout_keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_acc))

print(accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, dropout_keep_prob: 1.0}))
