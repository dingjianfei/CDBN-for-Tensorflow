# Copyright 2016 Sanghoon Yoon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convolutional Deep Belief Network for Tensorflow"""

# ## Build The Computation Graph
#
# **`TensorFlow`** programs are usually structured into a construction phase, that assembles a graph, and an execution phase that uses a session to execute ops in the graph.
#
# For example, it is common to create a graph to represent and train a neural network in the construction phase, and then repeatedly execute a set of training ops in the graph in the execution phase.

# ### Batch Image Inputs
# Decode JPEG images and convert them into tensors and batch them.

# In[7]:

import tensorflow as tf
import numpy as np

# Constants
FLAGS = tf.app.flags.FLAGS
FLAGS.CDBN = "CDBN/"

# Decoding
FLAGS.num_threads = 8

# Input structure
FLAGS.height = 300
FLAGS.width = 300
FLAGS.depth = 3
FLAGS.shapes = [FLAGS.height, FLAGS.width, FLAGS.depth]

# Training
FLAGS.num_epochs = 2
FLAGS.batch_size = 5
FLAGS.learning_rate = 0.03


def decode_file(filename_queue):
    reader = tf.WholeFileReader()
    filename, content = reader.read(filename_queue)
    example = tf.image.decode_jpeg(content)
    # Explicitly set shape
    example = tf.reshape(example, shape=FLAGS.shapes)

    return example, filename


def input_pipeline(filenames, batch_size, num_threads, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example_list = [decode_file(filename_queue) for _ in range(num_threads)]

    min_after_dequeue = 5000
    capacity = min_after_dequeue + (num_threads + 1) * batch_size
    example_batch, filename_batch = tf.train.shuffle_batch_join(
        example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, filename_batch


input = input_pipeline(photo_paths, FLAGS.batch_size, FLAGS.num_threads, num_epochs=FLAGS.num_epochs)

# ### Build Convolutional RBM
#
# Build the convolutional deep belief network(CDBN) to learn distributions of images.
#
# Before building the whole CDBN which is comprised of multiple convolutional restricted Boltzmann machines(CRBM) and multiple fully connected layers, we should build CRBM first. In CRBM, the weights between the hidden and visible layers are shared among all locations in an image.

# In[8]:

Nv = FLAGS.height
# Nh = 280
# Nw = Nv - Nh + 1
Nw = 10
K1 = 32
K2 = 64
c = 2


# Reusable variable generator
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(constant, shape, name):
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, c, c, 1],
                          strides=[1, c, c, 1], padding='SAME')


# Input placeholder (visual layer)
x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size] + FLAGS.shapes)
vis_0 = tf.div(x, 255)

# First convolutional rbm
W_conv0 = weight_variable([Nw, Nw, FLAGS.depth, K1], 'Weights0')
b_conv0 = bias_variable(0.0, [K1], 'Bias0')

# Probability to activate for hidden unit
h_conv0 = tf.sigmoid(conv2d(vis_0, W_conv0) + b_conv0)
# h_pool1 = max_pool(h_conv1)

# Sample hidden layer state (Bernoulli)
h_state0 = tf.nn.relu(tf.sign(h_conv0 - tf.random_uniform(h_conv0.get_shape(), maxval=1)))

# Variables for Gibbs sampling visual units
W_flipped = tf.transpose(tf.reverse(W_conv0, [True, True, False, False]), perm=[0, 1, 3, 2])
c_vis = bias_variable(0.0, (), 'Cias0')

# Gibbs sample visual layer
dist = tf.contrib.distributions.Normal(mu=conv2d(h_state0, W_flipped) + c_vis, sigma=1.)
vis_1 = tf.reshape(tf.div(dist.sample_n(1), Nw * Nw), [FLAGS.batch_size, Nv, Nv, FLAGS.depth])
# Gibbs sample hidden layer
h_conv1 = tf.sigmoid(conv2d(vis_1, W_conv0) + b_conv0)


def get_ith_element_4d(tensor, i, shape):
    return tf.slice(tensor, [i, 0, 0, 0], shape)


def get_ij_flat_vis_4d(tensor, i, j):
    flat_shape = [Nv, Nv]
    ith_vis = get_ith_element_4d(tensor, i, [1] + flat_shape + [FLAGS.depth])
    # Return [1, Nv, Nv, 1] shape
    return tf.slice(ith_vis, [0, 0, 0, j], [1] + flat_shape + [1])


def get_ith_hid_filter_4d(tensor, i):
    flat_shape = [Nv, Nv]
    ith_hid = get_ith_element_4d(tensor, i, [1] + flat_shape + [K1])
    # Return [Nv, Nv, 1, K1] shape
    return tf.reshape(ith_hid, flat_shape + [1, K1])


grad_batches = tf.zeros([Nw, Nw, FLAGS.depth, K1])
grad_bias = tf.zeros([K1])
grad_cias = 0.0
for idx in range(0, FLAGS.batch_size):
    h_prob0 = get_ith_hid_filter_4d(h_conv0, idx)
    h_prob1 = get_ith_hid_filter_4d(h_conv1, idx)
    grad_b = tf.reshape(tf.reduce_mean(h_prob0 - h_prob1, [0, 1]), [K1])
    grad_bias = tf.add(grad_bias, tf.mul(FLAGS.learning_rate, grad_b))

    grad_c = tf.reduce_mean(vis_0 - vis_1)
    grad_cias = tf.add(grad_cias, tf.mul(FLAGS.learning_rate, grad_c))

    hid_filter0 = tf.reverse(h_prob0, [True, True, False, False])
    hid_filter1 = tf.reverse(h_prob1, [True, True, False, False])
    positive = [0] * FLAGS.depth
    negative = [0] * FLAGS.depth
    one_ch_conv_shape = [1, Nv, Nv, 1, K1]
    for jdx in range(0, FLAGS.depth):
        positive[jdx] = tf.reshape(conv2d(get_ij_flat_vis_4d(vis_0, idx, jdx), hid_filter0), one_ch_conv_shape)
        negative[jdx] = tf.reshape(conv2d(get_ij_flat_vis_4d(vis_1, idx, jdx), hid_filter1), one_ch_conv_shape)
    positive = tf.concat(3, positive)
    negative = tf.concat(3, negative)
    cropped = tf.reshape(tf.slice(tf.sub(positive, negative), [0, 0, 0, 0, 0], [1, Nw, Nw, FLAGS.depth, K1]),
                         [Nw, Nw, FLAGS.depth, K1])
    grad_w = tf.div(cropped, Nv * Nv)
    grad_batches = tf.add(grad_batches, tf.mul(FLAGS.learning_rate, grad_w))

loss_func = tf.sqrt(tf.reduce_mean(tf.square(vis_0 - vis_1)))
tf.scalar_summary(FLAGS.CDBN + 'Loss Function', loss_func)
gradient_ascent = [W_conv0.assign_add(grad_batches), b_conv0.assign_add(grad_bias), c_vis.assign_add(grad_cias)]
tf.scalar_summary(FLAGS.CDBN + 'Gradient', tf.reduce_mean(grad_batches))

batch_num = tf.Variable(tf.constant(0))
increment_batch_num = batch_num.assign_add(1)
tf.scalar_summary(FLAGS.CDBN + 'Batch Num', batch_num)
tf.scalar_summary(FLAGS.CDBN + 'Weight', tf.sqrt(tf.reduce_mean(tf.square(W_conv0))))
tf.scalar_summary(FLAGS.CDBN + 'Bias', tf.sqrt(tf.reduce_mean(tf.square(b_conv0))))
tf.scalar_summary(FLAGS.CDBN + 'Cias', c_vis)

tf.image_summary(FLAGS.CDBN + 'Input Image', x, max_images=FLAGS.batch_size)
tf.image_summary(FLAGS.CDBN + 'Hidden Image',
                 tf.transpose(tf.reduce_mean(h_conv0, 0, keep_dims=True), perm=[3, 1, 2, 0]), max_images=K1)
tf.image_summary(FLAGS.CDBN + 'Generated Image', vis_1, max_images=FLAGS.batch_size)
tf.image_summary(FLAGS.CDBN + 'Weight Image', tf.transpose(W_conv0, perm=[3, 0, 1, 2]), max_images=K1)

# cost = tf.add(cost, tf.reduce_mean(grad0))

# # Second convolutional layer
# W_conv2 = weight_variable([Nw, Nw, K1, K2])
# b_conv2 = bias_variable([K2])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool(h_conv2)

# Nf = Nv / c / c
# No = 1024

# # Fully connected layer
# W_fc1 = weight_variable([Nf * Nf * K2, No])
# b_fc1 = bias_variable([No])

# h_pool2_flat = tf.reshape(h_pool2, [-1, Nf * Nf * K2])
# h_fc1 = tf.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# # Dropout to prevent overfitting
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# # Cost function
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# # Evaluation
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# sess.run(tf.initialize_all_variables())
# for i in range(20000):
#   batch = mnist.train.next_batch(50)
#   if i%100 == 0:
#     train_accuracy = accuracy.eval(feed_dict={
#         x:batch[0], y_: batch[1], keep_prob: 1.0})
#     print("step %d, training accuracy %g"%(i, train_accuracy))
#   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# ### Initialize All Variables
# Initialize all variables

# In[ ]:

merged_summary = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('summary')
# init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
init_op = tf.initialize_local_variables()

saver = tf.train.Saver()

# ## Launch The Graph in a Session
# Create a `Session` object and launch the operations in the graph.

# In[ ]:

with tf.Session() as sess:
    # with tf.InteractiveSession() as sess:
    saver.restore(sess, 'models/cdbn')
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        #     examples, filenames = sess.run(input)
        #     print examples[0]
        #     image = Image(filename=filenames[0])
        #     print sess.run(h_conv0, feed_dict={x: examples}).shape
        while not coord.should_stop():
            images, filenames = sess.run(input)
            ops = gradient_ascent + [increment_batch_num, merged_summary]
            weights, bias, cias, b_n, summary = sess.run(ops, feed_dict={x: images})
            summary_writer.add_summary(summary, b_n)
            saver.save(sess, 'models/cdbn')

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

