#!/usr/bin/env python3

import pygame
import pygame.midi
import os
import time
import magenta
import tensorflow as tf

from pygame.locals import *
from magenta.models.melody_rnn import melody_rnn_config_flags
from magenta.models.melody_rnn import melody_rnn_model
from magenta.models.melody_rnn import melody_rnn_sequence_generator

from magenta.music.protobuf import generator_pb2
from magenta.music.protobuf import music_pb2
import tensorflow.compat.v1 as tf

from enum import Enum


class State(Enum):
    DISPLAY = 1
    RECORD = 2
    GENERATE = 3
    PLAYBACK = 4


def next_state():
    global state
    j = state.value
    j = j + 1
    if j > len(State):
        state = State(1)
    else:
        state = State(j)
        print(state.name, state.value)
    return True


def introduction():
    global screen, font, small_font, nations, nation_idx, MSG1, MSG2, MSG3

    text1 = small_font.render(MSG1, True, (255, 255, 0))
    text2 = small_font.render(MSG2, True, (255, 255, 0))
    text3 = small_font.render(MSG3, True, (255, 255, 0))
    text4 = font.render(nations[nation_idx], True, (255, 0, 0))
    screen.blit(text1, (300, 200))
    screen.blit(text2, (300, 260))
    screen.blit(text3, (300, 320))
    screen.blit(text4, (750, 550))
    pygame.display.update()
    nation_idx = nation_idx + 1
    if nation_idx >= len(nations):
        next_state()
    return


def record_midi():
    global note_str, note_seq, font, screen
    if keyboard.poll():
        midi_events = keyboard.read(10)
        for event in midi_events:
            msg = event[0]
            status = msg[0]
            note = msg[1]
            velocity = msg[2]
            if status == NOTE_ON:
                note_str = note_str + NOTES[note % 12] + ' '
                if len(note_seq) < MAX_NOTES:
                    music.note_on(note, velocity)
                    note_seq.append(note)
            elif status == NOTE_OFF:
                music.note_off(note)

    msg = "Input your 5 notes"
    text1 = font.render(msg, True, (255, 255, 0))
    text2 = font.render(note_str, True, (255, 255, 0))
    screen.blit(text1, (300, 200))
    screen.blit(text2, (300, 300))
    pygame.display.update()
    if len(note_seq) >= MAX_NOTES:
        next_state()
    return


def make_melody():
    global keyboard, music, font, screen

    msg = "Generating your National Anthem"
    text1 = font.render(msg, True, (255, 255, 0))
    text2 = font.render(note_str, True, (255, 255, 0))
    screen.blit(text1, (300, 200))
    screen.blit(text2, (300, 300))
    pygame.display.update()

    prime = []
    for j in range(len(note_seq)):
        prime.append(note_seq[j])
        prime.append(-2)

    print(prime)
    primer_melody = magenta.music.Melody(prime)
    primer_sequence = primer_melody.to_sequence(qpm=qpm)

    generator_options = generator_pb2.GeneratorOptions()
    generator_options.args['temperature'].float_value = 1.0
    generator_options.args['beam_size'].int_value = 1
    generator_options.args['branch_factor'].int_value = 1
    generator_options.args['steps_per_iteration'].int_value = 1

    generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
        model=melody_rnn_model.MelodyRnnModel(config),
        details=config.details,
        steps_per_quarter=config.steps_per_quarter,
        checkpoint=os.path.join(run_dir, 'train'),
        bundle=None)

    seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
    total_seconds = num_steps * seconds_per_step

    if primer_sequence:
        input_sequence = primer_sequence
        if primer_sequence.notes:
            last_end_time = max(n.end_time for n in primer_sequence.notes)
        else:
            last_end_time = 0
        generate_section = generator_options.generate_sections.add(
            start_time=last_end_time + seconds_per_step,
            end_time=total_seconds)

        if generate_section.start_time >= generate_section.end_time:
            tf.logging.fatal(
                'Priming sequence is longer than the total number of steps '
                'requested: Priming sequence length: %s, Generation length '
                'requested: %s',
                generate_section.start_time, total_seconds)

    generated_sequence = generator.generate(input_sequence, generator_options)
    magenta.music.sequence_proto_to_midi_file(generated_sequence, song_file)

    keyboard.close()
    music.close()
    del keyboard
    del music
    pygame.mixer.music.load(song_file)
    pygame.mixer.music.play()
    next_state()
    return


def play_back():
    global font, screen

    pos = pygame.mixer.music.get_pos()/1000
    pos = round(17-pos, 1)
    msg = "Playing your National Anthem"
    text1 = font.render(msg, True, (255, 255, 0))
    text2 = font.render(str(pos).rjust(4), True, (255, 255, 0))
    screen.blit(text1, (300, 200))
    screen.blit(text2, (300, 300))
    pygame.display.update()
    return


FPS = 60
NOTE_ON = 144
NOTE_OFF = 128
MAX_NOTES = 5
PIANO = 0
WINDOW = [1920, 1080]
NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
SONG_END = pygame.USEREVENT + 1
MSG1 = 'National Anthem is an  artificial intelligence artwork to generate a new'
MSG2 = 'national anthem by learning the melodies of the 140 national anthems and'
MSG3 = 'also with your five input notes as initial melody.'

data_dir = os.path.join(os.getcwd(), 'data')
run_dir = 'logdir'
song_file = data_dir + os.path.sep + 'MyMelody.mid'
back_file = data_dir + os.path.sep + 'Background04.jpg'
nations_file = data_dir + os.path.sep + 'Nations.txt'

tf.logging.set_verbosity('INFO')
config = melody_rnn_model.default_configs['attention_rnn']
config.hparams.batch_size = 64
config.hparams.rnn_layer_sizes = [64, 64]
qpm = magenta.music.DEFAULT_QUARTERS_PER_MINUTE
num_steps = 128

pygame.init()
pygame.midi.init()
pygame.font.init()

screen = pygame.display.set_mode(WINDOW, pygame.FULLSCREEN | pygame.HWSURFACE)
# screen = pygame.display.set_mode(WINDOW, pygame.HWSURFACE)
pygame.display.set_caption('Music')
pygame.mixer.music.set_endevent(SONG_END)
font = pygame.font.SysFont('courier', 48)
small_font = pygame.font.SysFont('Courier', 30)
clock = pygame.time.Clock()
pygame.mouse.set_visible(False)
background = pygame.image.load(back_file)
running = True
state = State(1)

for i in range(pygame.midi.get_count()):
    dev = pygame.midi.get_device_info(i)
    print(dev)

keyboard = pygame.midi.Input(1)
music = pygame.midi.Output(2)
# music = pygame.midi.Output(4) # OSX FluidSynth

music.set_instrument(PIANO)
note_str = ""
note_seq = []

input_file = open(nations_file, "r")
nations = []
nation_idx = 0
for line in input_file:
    line = line.strip().replace('\n', '')
    line = line.replace('.mid', '')
    line = line.replace('_', ' ')
    nations.append(line)


while running:
    screen.fill((0, 0, 0))
    screen.blit(background, (0, 0))

    if state == State.DISPLAY:
        introduction()
    elif state == State.RECORD:
        record_midi()
    elif state == State.GENERATE:
        make_melody()
    elif state == State.PLAYBACK:
        play_back()

    pygame.display.update()
    clock.tick(FPS)

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            elif e.key == pygame.K_SPACE:
                running = next_state()
        elif e.type == SONG_END:
            keyboard = pygame.midi.Input(1)
            music = pygame.midi.Output(2)
            # music = pygame.midi.Output(4) # OSX FluidSynth
            music.set_instrument(PIANO)
            note_str = ""
            note_seq = []
            nation_idx = 0
            running = next_state()


if keyboard is not None:
    keyboard.close()
    del keyboard
if music is not None:
    music.close()
    del music

pygame.mouse.set_visible(True)
pygame.mixer.music.stop()
pygame.midi.quit()
pygame.quit()
