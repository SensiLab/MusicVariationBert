
import torch
import torch.functional as F
import fairseq
import numpy as np

from typing import Tuple

BAR_START = "<0-0>"
BAR_END = "<0-255>"

POS_START = "<1-0>"
POS_END = "<1-127>"

INS_START = "<2-0>"
INS_END = "<2-127>"

PITCH_START = "<3-0>"
PITCH_END = "<3-255>"

MIN_PITCH = 517 # C-2
MAX_PITCH = 644 # G8

DUR_START = "<4-0>"
DUR_END = "<4-127>"

VEL_START = "<5-0>"
VEL_END = "<5-31>"

SIG_START = "<6-0>"
SIG_END = "<6-253>"

TEMPO_START = "<7-0>"
TEMPO_END = "<7-48>"

SPECIAL_TOKENS = ['<mask>', '<s>', '<pad>', '</s>', '<unk>']

NOTES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

NOTE_TO_TOKEN = {
    "C": 517,
    "C#": 518,
    "Db": 518,  # Enharmonic equivalent of C#
    "D": 519,
    "D#": 520,
    "Eb": 520,  # Enharmonic equivalent of D#
    "E": 521,
    "F": 522,  
    "F#": 523,
    "Gb": 523,  # Enharmonic equivalent of F#
    "G": 524,
    "G#": 525,
    "Ab": 525,  # Enharmonic equivalent of G#
    "A": 526,
    "A#": 527,
    "Bb": 527,  # Enharmonic equivalent of A#
    "B": 528
}

# not always accurate in convention but practicle for this program
KEYS = {
    "C Major": ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
    "C# Major": ['C#', 'D#', 'F', 'F#', 'G#', 'A#', 'C'],
    "D Major": ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
    "E Major": ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
    "F Major": ['F', 'G', 'A', 'Bb', 'C', 'D', 'E'],
    "F# Major": ['F#', 'G#', 'A#', 'B', 'C#', 'D#', 'F'],
    "G Major": ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
    "A Major": ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
    "B Major": ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#'],
    "Cb Major": ['B', 'Db', 'Eb', 'E', 'Gb', 'Ab', 'Bb'],
    "Db Major": ['Db', 'Eb', 'F', 'Gb', 'Ab', 'Bb', 'C'],
    "Eb Major": ['Eb', 'F', 'G', 'Ab', 'Bb', 'C', 'D'],
    "Gb Major": ['Gb', 'Ab', 'Bb', 'B', 'Db', 'Eb', 'F'],
    "Ab Major": ['Ab', 'Bb', 'C', 'Db', 'Eb', 'F', 'G'],
    "Bb Major": ['Bb', 'C', 'D', 'Eb', 'F', 'G', 'A'],
    "A Minor": ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    "A# Minor": ['A#', 'C', 'C#', 'D#', 'F', 'F#', 'G#'],
    "B Minor": ['B', 'C#', 'D', 'E', 'F#', 'G', 'A'],
    "C Minor": ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb'],
    "C# Minor": ['C#', 'D#', 'E', 'F#', 'G#', 'A', 'B'],
    "D Minor": ['D', 'E', 'F', 'G', 'A', 'Bb', 'C'],
    "D# Minor": ['D#', 'F', 'F#', 'G#', 'A#', 'B', 'C#'],
    "E Minor": ['E', 'F#', 'G', 'A', 'B', 'C', 'D'],
    "F Minor": ['F', 'G', 'Ab', 'Bb', 'C', 'Db', 'Eb'],
    "F# Minor": ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E'],
    "G Minor": ['G', 'A', 'Bb', 'C', 'D', 'Eb', 'F'],
    "G# Minor": ['G#', 'A#', 'B', 'C#', 'D#', 'E', 'F#'],
    "Bb Minor": ['Bb', 'C', 'Db', 'Eb', 'F', 'Gb', 'Ab'],
    "Eb Minor": ['Eb', 'F', 'Gb', 'Ab', 'Bb', 'B', 'Db'],
    "Ab Minor": ['Ab', 'Bb', 'B', 'Db', 'Eb', 'E', 'Gb']
}

def bar_range(label_dict): return label_dict.index(BAR_START), label_dict.index(BAR_END)+1
def pos_range(label_dict): return label_dict.index(POS_START), label_dict.index(POS_END)+1
def ins_range(label_dict): return label_dict.index(INS_START), label_dict.index(INS_END)+1
def pitch_range(label_dict): return label_dict.index(PITCH_START), label_dict.index(PITCH_END)+1
def dur_range(label_dict): return label_dict.index(DUR_START), label_dict.index(DUR_END)+1
def vel_range(label_dict): return label_dict.index(VEL_START), label_dict.index(VEL_END)+1
def sig_range(label_dict): return label_dict.index(SIG_START), label_dict.index(SIG_END)+1
def tempo_range(label_dict): return label_dict.index(TEMPO_START), label_dict.index(TEMPO_END)+1

#Returns dictionary with keys = token_ids & values = token strings
def reverse_label_dict(label_dict: fairseq.data.dictionary.Dictionary):
  '''
  Midiformers code
  '''
  return {v: k for k, v in label_dict.indices.items()}

def get_key_notes(key: str) -> Tuple[list, list]:
    """
    Function takes a key as input and returns the list of
    notes in that key,
    @author: Stephen Krol

    Args:
        key: key of the scale, consists of a root note
            followed by either major or minor.
    Returns:
        list: list of notes in the key.
        list: list of notes not in key.
    """

    if key is None:
      return None, None
    
    assert key in KEYS, f'{key} not valid.'

    possible_notes = np.array(range(517, 529))

    key_notes = [NOTE_TO_TOKEN[note] for note in KEYS[key]]
    chromatic_notes = np.setdiff1d(possible_notes, key_notes)
    all_chromatic_notes = []

    # -2 -> 8
    for i in range(11):
      for note in chromatic_notes:
        all_chromatic_notes.append(note + i*12)
    
    all_key_notes = []

    for i in range(11):
      for note in key_notes:
         all_key_notes.append(note + i*12)

    return all_key_notes, all_chromatic_notes

def filter_invalid_indexes(logits, 
                           prev_index, 
                           label_dict, 
                           rev_inv_map, 
                           filtered_pitches_idx:list,
                           invalid_bars:list=None,
                           filter_value=-float('Inf')):
  """ Filter a distribution of logits using prev_predicted token 
        Args:
            logits: logits distribution shape (vocabulary size)
            prev_index: previous predicted token 
            label_dict : dictionary mapping string octuple encodings to indices 
            filtered_pitches_idx (list): indexes of pitches to be filtered
            invalid_bars (list): indexes for invalid bars
      Returns: filtered logits according to prev_idx 

      @author: midiformers
      @modified: sjkrol
  """
  
  logits = logits.clone()
    
  prev_index = prev_index.item()
  str_encoding = rev_inv_map[prev_index]

  # For example if previous index was pitch than according to Octuple encoding next note should be duration 
  # Therefore we fill up all the other 7 element ranges with infinity
  
  for tok in SPECIAL_TOKENS:
      logits[label_dict.index(tok)] = filter_value

  # if previous token was 'bar' then we mask everything excluding 'pos' 
  if(str_encoding[1] == '0'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value
  # pos
  elif(str_encoding[1] == '1'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value
  # ins
  elif(str_encoding[1] == '2'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value

    # filter pitches based off specified range
    if len(filtered_pitches_idx) != 0:    
      logits[filtered_pitches_idx] = filter_value
    
    # filter out chromatic pitches
    # if key is not None:
    #   key_notes, chromatic_notes = get_key_notes(key)
      # logits[chromatic_notes] = filter_value
      # chormatic_logits = logits[chromatic_notes]
      # key_logits = logits[key_notes]
    
    # filter out percussive pitches
    percussion_idxs = [i for i in range(MAX_PITCH+1, MAX_PITCH+129)]
    logits[percussion_idxs] = filter_value

    # probs = torch.softmax(logits, dim=-1)
    # key_probs = probs[key_notes]
    # chromatic_probs = probs[chromatic_notes]
    # pitch_probs = probs[list(range(*pitch_range(label_dict)))]
    # top5 = np.argpartition(key_probs, -5)[-5:]
    # print(f'Keys: {np.sort(key_probs[top5])}')
    # top5 = np.argpartition(chromatic_probs, -5)[-5:]
    # print(f'Chromatics: {np.sort(chromatic_probs[top5])}')

  # pitch
  elif(str_encoding[1] == '3'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value
  # dur
  elif(str_encoding[1] == '4'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value
  # vel
  elif(str_encoding[1] == '5'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value
  # sig
  elif(str_encoding[1] == '6'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
  # tempo
  # TODO: include after starting token
  elif(str_encoding[1] == '7'):
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value

    # filter invalid bars
    if invalid_bars is not None:
      logits[invalid_bars] = filter_value
  
  return logits 

def top_k_top_p(logits_batch, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """

    # TODO: is this really needed?
    logits_batch = logits_batch.clone()

    # print(logits_batch.dim())

    if(logits_batch.dim() == 1):
      logits_batch = logits_batch.unsqueeze(0)

    assert logits_batch.dim() == 2  # batch size 1 for now - could be updated for more but the code would be less clear
    
    # iterate through batch size 
    for index, logits in enumerate(logits_batch):
      top_k = min(top_k, logits.size(-1))  # Safety check
      if top_k > 0:
          # Remove all tokens with a probability less than the last token of the top-k
          indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
          logits[indices_to_remove] = filter_value

      if top_p > 0.0:
          sorted_logits, sorted_indices = torch.sort(logits, descending=True)
          cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

          # Remove tokens with cumulative probability above the threshold
          sorted_indices_to_remove = cumulative_probs > top_p
          # Shift the indices to the right to keep also the first token above the threshold
          sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
          sorted_indices_to_remove[..., 0] = 0

          indices_to_remove = sorted_indices[sorted_indices_to_remove]
          logits[indices_to_remove] = filter_value
    return logits_batch

def switch_temperature(prev_index: int, reverse_dict, temperature_dict):
  """ Changes temperature to value for one of the eight fields in octuple 
      Args: 
        logits: logits distribution shape (vocabulary size)
        prev_index: previous predicted token 
        label_dict : dictionary mapping string octuple encodings to indices 
        temperature_dict : dict containing temperature values for all the 8 individual octuple elements 
      Returns: next temperature value 

      @author: midiformers
      @modified: sjkrol
  """

  # First we convert the token to it's string mapping 
  prev_index = prev_index.item()
  str_encoding = reverse_dict[prev_index]

  if str_encoding[1] == 's':
     return temperature_dict[0]
  else:
    return temperature_dict[((int(str_encoding[1]) + 1)%(8))] 

def decode_w_label_dict(rev_inv_map: dict, octuple_midi_enc:torch.Tensor):
  '''
  Midiformers code
  @modified: sjkrol
  '''
  octuple_midi_enc_copy = octuple_midi_enc.clone().tolist()
  seq = []
  for token in octuple_midi_enc_copy:
    seq.append(rev_inv_map[token])

  seq_str = " ".join(seq)

  del octuple_midi_enc_copy
  return seq_str

def random_masking(encoding, note_percentage_random_mask, label_dict, mask_inst=False, mask_dur=False, mask_bar=False):

    mask_idx = label_dict.index('<mask>')
    masked_octs = np.random.choice( a = range(1, int( len(encoding) / 8 ) - 2) , \
                            size = int( (len(encoding) / 8) * (note_percentage_random_mask/100) ), \
                            replace = False)

    if not mask_inst and not mask_dur:
        for masked_oct in masked_octs:
            encoding.index_fill_(0, torch.tensor( masked_oct * 8 + [1,3,5,6,7] ), mask_idx)
    elif not mask_inst:
        for masked_oct in masked_octs:
            encoding.index_fill_(0, torch.tensor( masked_oct * 8 + [0,1,3,4,5,6,7] ), mask_idx)
    else:
        for masked_oct in masked_octs:
            encoding[ masked_oct * 8: (masked_oct + 1)*8 ] = mask_idx
    
    return encoding