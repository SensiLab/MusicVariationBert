
import os

from MusicVariationBert.generation import generate_variations, write_variation
from MusicVariationBert.preprocess import gen_dictionary
from MusicVariationBert.utils import reverse_label_dict
from MusicVariationBert.musicbert import MusicBERTModel

NOTES = ['N/A','C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
NOTES_TOKEN = {note : token for note, token in zip(NOTES[1:], list(range(541, 553)))}

def convert_to_encoding(note: str, octave: int):
    """
    Converts a note to its model encoding.
    """

    return NOTES_TOKEN[note] + 12*octave

if not os.path.exists('input0/dict.txt'):
    os.mkdir("input0")
    gen_dictionary("input0/dict.txt")

if not os.path.exists('label/dict.txt'):
    os.mkdir("label")
    gen_dictionary("label/dict.txt")

# load model
roberta_base = MusicBERTModel.from_pretrained('.', 
   #  checkpoint_file='FILEPATH TO MODEL/checkpoint_last_musicbert_base_w_genre_head.pt' # update path to point to your model
   checkpoint_file='../RhapsodyRefiner/static/Weights/checkpoint_last_musicbert_base_w_genre_head.pt'
)


# extract label dict and create reverse label dict
label_dict = roberta_base.task.label_dictionary
reversed_dict = reverse_label_dict(label_dict)

roberta_base.eval()

### ARGUMENTS ###
filename = "Pitch Test.mid" # filepath to midi file 
n_var=1 # number of variations to generate
variation_percentage=20 # how many notes to vary
new_notes=True # set true if you want to add new notes
new_notes_percentage=10 # percentage of new notes to add, in relation to total notes in the midi file

"""
Position: where a midi note's start time is
Pitch: pitch of the midi notes
Duration: how long the note plays for
Velocity: how the note is played
"""
attributes = ["Position", "Pitch", "Duration", "Velocity"]
temperature_dict = {
    "Bar" : 1,
    "Position" : 1, # position
    "Instrument" : 1,
    "Pitch" : 1, # pitch
    "Duration" : 1, # duration
    "Velocity" : 1, # velocity
    "Time signature" : 1,
    "Tempo" : 1,
}

# pitch range to vary over
# notes range from C0 -> Ab9
# can be none to give model full freedom
# can assign both values, however, max pitch must be greater than min pitch
min_pitch = convert_to_encoding(note="C", octave=4) # C4
max_pitch = None

key = "C Major" # can be none, giving the model more freedom
"""
"C Major"     "A Minor"
"C# Major"    "A# Minor"
"D Major"    "B Minor"
"E Major"    "C Minor"
"F Major"    "C# Minor"
"F# Major"    "D Minor"
"G Major"    "D# Minor"
"A Major"    "E Minor"
"B Major"    "F Minor"
"Cb Major"    "F# Minor"
"Db Major"    "G Minor"
"Eb Major"    "G# Minor"
"Gb Major"    "Bb Minor"
"Ab Major"    "Eb Minor"
"Bb Major"    "Ab Minor"
"""
# beta controls liklihood of predicting a chromatic note given a key
# if key is None beta does nothing
# rand [0, 1], 0 means no chromatic notes, 1 only chromatic notes are predicted
beta = 0

# bars
# bars to run variation over, operates over [0, MAX-BAR-IN-TRACK-1]
# multiple ranges can be provided, e.g. [(0, 0), (1, 5), (9, 10)] = Only vary bars 0, 1, 2, 3, 4, 5, 9, 10
# make sure there is no overlap between bars
bars = None


#### GENERATE VARIATIONS ####
variations = generate_variations(filename=filename, 
                                n_var=n_var, 
                                roberta_base=roberta_base,
                                label_dict=label_dict,
                                reversed_dict=reversed_dict,
                                new_notes=new_notes,
                                new_notes_percentage=new_notes_percentage,
                                variation_percentage=variation_percentage,
                                attributes=attributes, 
                                temperature_dict=temperature_dict, 
                                min_pitch=min_pitch,
                                max_pitch=max_pitch,
                                key=key,
                                beta=beta,
                                bars=None,
                                multinomial_sample=True)

write_variation(variations[0], "ENTER NAME.mid", reversed_dict) # write one variation, 