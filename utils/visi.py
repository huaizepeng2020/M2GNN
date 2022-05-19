# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import getopt
import numpy as np
import tensorflow as tf

if tf.__version__[0] == '1':
    from tensorflow.contrib.tensorboard.plugins import projector
else:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    from tensorboard.plugins import projector

tf.logging.set_verbosity(tf.logging.INFO)


class Visualizer(object):
    def __init__(self, vecPath, metaPath, logdir):
        self.vec_path = os.path.abspath(vecPath)
        self.meta_path = os.path.abspath(metaPath)
        self.log_dir = os.path.abspath(logdir)
        tf.logging.info("vec_path:{}".format(self.vec_path))
        tf.logging.info("meta_path:{}".format(self.meta_path))
        tf.logging.info("log_dir:{}".format(self.log_dir))

    def run(self):
        tf.logging.info('-------start visualize-------')

        vec = tf.Variable(self.read_vec(), dtype=tf.float32, name='vector')
        tf.logging.info("vec: %s" % vec)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.log_dir, "model.ckpt"), global_step=0)
        summary_writer = tf.summary.FileWriter(self.log_dir)

        config = projector.ProjectorConfig()
        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = vec.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = self.meta_path
        # Saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)

        tf.logging.info('-------end visualize-------')

    def read_vec(self):
        # 行号从0开始，区间前闭后开
        line_num = 0
        vecs = []
        sep = '\t'
        with tf.gfile.Open(self.vec_path, 'r') as read_in:
            for line in read_in:
                if line_num == 0:
                    if ',' in line:
                        sep = ','
                    elif ';' in line:
                        sep = ';'
                    elif ' ' in line:
                        sep = ' '
                    elif '\t' in line:
                        sep = '\t'
                vec = list(map(lambda x: float(x), line.strip().split(sep)))
                vecs.append(vec)
                line_num += 1
        tf.logging.info('line_num: %s' % line_num)
        tf.logging.info('vecs: %s' % vecs[0])  # 如果提示map不匹配float，需要切换到py27
        return vecs


def main(vecPath, metaPath, logdir):
    visualizer = Visualizer(vecPath, metaPath, logdir)
    visualizer.run()


if __name__ == "__main__":
    usage = "xxx.python -v vecPath -m metaPath -l logdir"
    vecPath = "";
    metaPath = "";
    logdir = ""
    opts, _ = getopt.getopt(sys.argv[1:], "v:m:l:")
    print(opts)
    for opt, value in opts:
        if opt == "-v":
            vecPath = value
        elif opt == "-m":
            metaPath = value
        elif opt == "-l":
            logdir = value
    if vecPath == "" or metaPath == "" or logdir == "":
        print(usage)
        exit(-1)

    # vecPath = 'ad_vec'
    # metaPath = 'ad_meta'
    # logdir = './'

    main(vecPath, metaPath, logdir)
