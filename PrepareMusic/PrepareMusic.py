#!/usr/bin/env python3

import os
from magenta.music import midi_io
from magenta.music import note_sequence_io
from magenta.models.melody_rnn import melody_rnn_config_flags
from magenta.models.melody_rnn import melody_rnn_pipeline
from magenta.models.melody_rnn import melody_rnn_model
from magenta.pipelines import pipeline
import tensorflow as tf

tf.logging.set_verbosity('INFO')

work_dir = os.getcwd()
input_dir = os.path.join(work_dir, 'data')
output_dir = os.path.join(work_dir, 'output')
anthems_file = os.path.join(output_dir, 'anthems.tfrecord')

# files_in_dir = tf.gfile.ListDirectory(input_dir)
files_in_dir = tf.io.gfile.listdir(input_dir)

with note_sequence_io.NoteSequenceRecordWriter(anthems_file) as writer:

    for file_in_dir in files_in_dir:
        full_file_path = os.path.join(input_dir, file_in_dir)
        print(full_file_path)
        try:
            sequence = midi_io.midi_to_sequence_proto(
                tf.io.gfile.GFile(full_file_path, 'rb').read())
        except midi_io.MIDIConversionError as e:
            tf.logging.warning(
                'Could not parse midi file %s. Error was: %s',
                full_file_path, e)

        sequence.collection_name = os.path.basename(work_dir)
        sequence.filename = os.path.join(output_dir, os.path.basename(full_file_path))
        sequence.id = note_sequence_io.generate_note_sequence_id(sequence.filename, sequence.collection_name, 'midi')

        if sequence:
            writer.write(sequence)

filenames = [anthems_file]
dataset = tf.data.TFRecordDataset(filenames)

config = melody_rnn_model.default_configs['attention_rnn']
pipeline_instance = melody_rnn_pipeline.get_pipeline(config, eval_ratio=0.0)

pipeline.run_pipeline_serial(
    pipeline_instance,
    pipeline.tf_record_iterator(anthems_file, pipeline_instance.input_type),
    output_dir)
