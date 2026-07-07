from sqlite3 import converters

import numpy as np
from music21 import corpus, instrument, note, chord, stream
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

print("--- 1. Loading built-in Music Data... ---")
notes = []

import glob # Ye upar import hona chahiye agar nahi hai toh

print("--- 1. Loading Hindi MIDI files... ---")
notes = []

# Aapke 'dataset' folder se saare hindi gaane load karne ke liye
midi_files = glob.glob("dataset/*.mid")

if len(midi_files) == 0:
    print("place your Hindi MIDI files in the 'dataset' folder and try again.")
else:
    for file in midi_files:
        try:
            # Aapke code me kuch aisa hoga, use temporary aisa badalye:
try:
    # jahan file parse ho rahi hai
    midi = converters.parse(file) 
except Exception as e:
    print(f"Error parsing {file}, skipping...")
    print(f"Actual Error: {e}")  # Ye line add karein taaki asli wajah pata chale
            midi = corpus.parse(file) if 'corpus' in dir() else converters.parse(file)
            notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Error parsing {file}, skipping...")

# Vocab taiyar karna
pitches = sorted(list(set(notes)))
n_vocab = len(pitches)
note_to_int = dict((note, number) for number, note in enumerate(pitches))

# Choti sequence length taki turant run ho jaye
sequence_length = 16 
network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(n_vocab)
network_output = to_categorical(network_output, num_classes=n_vocab)

print("--- 2. Building LSTM Model... ---")
model = Sequential()
model.add(LSTM(128, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print("--- 3. Training Model Fast (5 Epochs)... ---")
model.fit(network_input, network_output, epochs=5, batch_size=64)

print("--- 4. Generating New Music... ---")
start = np.random.randint(0, len(network_input)-1)
int_to_note = dict((number, note) for number, note in enumerate(pitches))
pattern = list(np.reshape(network_input[start] * n_vocab, (sequence_length,)).astype(int))
prediction_output = []

for note_index in range(100): # 100 notes generate honge
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    pattern.append(index)
    pattern = pattern[1:]

print("--- 5. Saving output_music.mid... ---")
offset = 0
output_notes = []
for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = [note.Note(int(current_note)) for current_note in notes_in_chord]
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        output_notes.append(new_note)
    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='output_music.mid')
print("🎉 Success! Project complete. 'output_music.mid' file ban gayi hai!")