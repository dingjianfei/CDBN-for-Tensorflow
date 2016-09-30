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

"""Tensorflow Runner"""

import os
import tensorflow as tf
from model import Model

flags = tf.app.flags
flags.DEFINE_string('summary_dir', 'summary', 'Directory where all event data are stored')
flags.DEFINE_string('model_path', 'models', 'File path where all models are stored')
flags.DEFINE_integer('num_threads', 24, 'The number of threads for running')
flags.DEFINE_integer('num_epochs', 3, 'The number of epochs')
flags.DEFINE_integer('batch_size', 32, 'The number of examples per batch')
flags.DEFINE_float('learning_rate', .01, 'Learning rate')

FLAGS = flags.FLAGS


class Runner:
    def __init__(self, preparator, model):
        if issubclass(model, Model):
            raise TypeError

        file_paths = preparator.prepare()

        def decode_file(filename_queue):
            reader = tf.WholeFileReader()
            filename, content = reader.read(filename_queue)
            example = tf.image.decode_jpeg(content)
            # Explicitly set shape
            # TODO: replace shape array with arguments or something
            example = tf.reshape(example, shape=[300, 300, 3])

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

        self.input_pipeline = input_pipeline(
            file_paths, FLAGS.batch_size, FLAGS.num_threads, num_epochs=FLAGS.num_epochs)

        self.model = model
        self.model.build_graphs()

        self.batch_num = tf.Variable(tf.constant(0))
        increment_batch_num = self.batch_num.assign_add(1)

        merged_summary = tf.merge_all_summaries()

        self.model.build_init_ops()
        self.model_path = '{}/batch_num'.format(FLAGS.model_path)
        if self.model_exists:
            self.init_op = tf.initialize_local_variables()
        else:
            self.init_op = [tf.initialize_local_variables(), tf.initialize_variables([self.batch_num])]
        self.saver = tf.train.Saver([self.batch_num])

        self.ops = [increment_batch_num, merged_summary]

    @property
    def model_exists(self):
        return os.path.exists(self.model_path)

    def run(self):
        with tf.Session() as sess:
            summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir, sess.graph)
            self.model.init_variables(sess)
            if self.model_exists:
                self.saver.restore(sess, self.model_path)
            sess.run(self.init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop():
                    images, filenames = sess.run(self.input_pipeline)
                    if self.model.ops is not None:
                        # Continue training
                        ops = self.model.ops + self.ops
                        results = sess.run(ops, feed_dict={self.model.input: images})
                        summary_writer.add_summary(results[-1], results[-2])
                        self.model.propagate_results(results[0:-2])
                        self.model.save(sess)
                        self.saver.save(sess, self.model_path)
                    else:
                        # Training finished
                        results = sess.run(self.model.output, feed_dict={self.model.input: images})

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')

            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()
