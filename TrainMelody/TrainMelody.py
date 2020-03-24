#!/usr/bin/env python3

import os
import magenta
from magenta.models.melody_rnn import melody_rnn_config_flags
from magenta.models.shared import events_rnn_graph
from magenta.models.shared import events_rnn_train
from magenta.models.melody_rnn import melody_rnn_model
import tensorflow.compat.v1 as tf

work_dir = os.getcwd()
# run_dir = os.path.join(work_dir, 'logdir')
run_dir = 'logdir'
print(run_dir)

tf.logging.set_verbosity('INFO')

sequence_example_file = 'data' + os.path.sep + 'training_melodies.tfrecord'
sequence_example_file_paths = tf.io.gfile.glob(os.path.join(work_dir, sequence_example_file))

config = melody_rnn_model.default_configs['attention_rnn']
config.hparams.batch_size = 64
config.hparams.rnn_layer_sizes = [64, 64]

mode = 'train'
build_graph_fn = events_rnn_graph.get_build_graph_fn(mode, config, sequence_example_file_paths)

train_dir = os.path.join(run_dir, 'train')
eval_dir = os.path.join(run_dir, 'eval')

print('Train directory: %s', train_dir)
print('Evaluate directory: %s', eval_dir)

print(config.hparams.batch_size)
print(magenta.common.count_records(sequence_example_file_paths))

# num_batches = magenta.common.count_records(sequence_example_file_paths) // config.hparams.batch_size

events_rnn_train.run_training(build_graph_fn, train_dir, 20000, 10, 3)
