import input_data
import tensorflow as tf

# A single hidden layer MLP

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# print(mnist.train.num_examples)
input_layer_size = 784
hidden_layer_size = 100
output_layer_size = 10

# First, define Graph inputs
x = tf.placeholder("float", [None, input_layer_size])
y = tf.placeholder("float", [None, output_layer_size])

W1 = tf.Variable(tf.random_normal([hidden_layer_size, input_layer_size]))
b1 = tf.Variable(tf.zeros([hidden_layer_size]))

W2 = tf.Variable(tf.random_normal([output_layer_size, hidden_layer_size]))
b2 = tf.Variable(tf.zeros([output_layer_size]))

h1 = tf.sigmoid(tf.matmul(x, W1, transpose_b=True) + b1)

output = tf.nn.softmax(tf.matmul(h1, W2, transpose_b=True) + b2)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), reduction_indices=1))

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

batch_size = 100
with tf.Session() as sess:

    sess.run(init)

    for epoch in range(20):
        tot_cost = 0.
        num_batches = int(mnist.train.num_examples / batch_size)
        for i in range(num_batches):
            x_batch, y_batch = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict={x: x_batch, y: y_batch})

            tot_cost += sess.run(cost, feed_dict={x: x_batch, y: y_batch})

        avg_cost = tot_cost / num_batches
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!!")

    corr_pred_vec = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(corr_pred_vec, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
