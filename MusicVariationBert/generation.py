'''
Inspired by: https://github.com/tripathiarpan20/midiformers
'''

import os
import torch
import fairseq
import numpy as np
from miditoolkit import MidiFile
from tqdm import tqdm
from typing import Tuple

from MusicVariationBert.preprocess import MIDI_to_encoding, encoding_to_MIDI, encoding_to_str, str_to_encoding, gen_dictionary
from MusicVariationBert.utils import (reverse_label_dict, 
                                      filter_invalid_indexes, 
                                      top_k_top_p, switch_temperature, 
                                      decode_w_label_dict, 
                                      pitch_range, get_key_notes)
from MusicVariationBert.musicbert import MusicBERTModel

# TODO: consonnance and disonnance
# TODO: fill empty bars (current)
# TODO: new notes when no notes in song

ATTRIBUTE_INDEXES = {"Bar" : 0,
                     "Position" : 1,
                     "Instrument" : 2,
                     "Pitch" : 3,
                     "Duration" : 4,
                     "Velocity" : 5,
                     "Time signature": 6,
                     "Tempo" : 7}

# encoded pitch value for non percussian notes
MIN_ENCODED_PITCH = 517 # caluclated 515, however C4 (60) is 589 : 589-60 = 529
MAX_ENCODED_PITCH = 643

def encode_midi_for_musicbert(filename: str, label_dict: dict):
    '''
    @author: sjkrol

    Function takes a midi filepath as input and encodes it in the correct
    format for MusicBERT to use as input.

    Args:
        midifile (str): filepath to midifile
        label_dict (dict): a dictionary that converts string events into their
            numeric counterpart.
    Returns:
        torch.tensor: a tensor cotaining a sequence of numbers representing the
            encoded midi file.
    '''

    midi_obj = MidiFile(filename)
    octuple_encoding = MIDI_to_encoding(midi_obj=midi_obj)
    encoding_str = encoding_to_str(octuple_encoding)
    encoding = label_dict.encode_line(encoding_str)

    return encoding

# TODO: test encoding to decoding consitency
def decode_musicbert_to_midi(encoding,reversed_dict, output_filepath):
    '''
    @author: sjkrol

    Function takes an encoding for the MusicBERT model and decodes
    it into a midi file.

    Args:
        encoding (torch.tensor): an encoding tensor for the MusicBERT
            model.
        label_dict (dict): a dictionary that converts string events into their
            numeric counterpart.
        output_filepath (str): the filepath to save the midi file
    Returns:
        None
    '''
    
    encoding_str = decode_w_label_dict(reversed_dict, encoding)
    octuple_encoding = str_to_encoding(encoding_str)
    midi_obj = encoding_to_MIDI(encoding=octuple_encoding)
    midi_obj.dump(output_filepath)

def bar_order_check(bars):
    '''
    Function takes a list of bars and checks they are in order. Bars can be a single integer
    or a tuple contaning a bar range.
    @author: sjkrol

    Args:
        bars (list): a list of bars, either single bar integers or bar ranges as tuples.
    Returns:
        bool: True if bar order is correct, False if not
    '''

    if bars is None:
        return True

    latest_bar = -1
    for bar in bars:
        # bar range
        if type(bar) is tuple:
            assert bar[0] <= bar[1], f"First bar in range should be less than last bar: {bar}"
            if bar[0] <= latest_bar:
                return False
            else:
                latest_bar = bar[1]
        # single bar
        else:
            if bar <= latest_bar:
                return False
            else:
                latest_bar = bar

    return True

def attributes_check(attributes: list):
    '''
    Function checks if attributes in the attribute list are of the correct type and value and that there are no duplicates. If not,
    the function will raise a ValueError or TypeError.
    @author: sjkrol

    Args:
        attributes (list): a list of attributes to mask. All values should be integers ranging between 0-7.
    Returns:
        None
    '''

    # used to check for duplicates
    found_attributes = {}

    for attribute in attributes:
        if attribute < 0 or attribute > 7:
            raise ValueError(f'Attribute {attribute} is not within the valid attribute range 0 - 7')
        elif type(attribute) is not int:
            raise TypeError(f"Attribute values must be of type int, however attribute {attribute} is of type {type(attribute)}")
        elif attribute in found_attributes:
            raise ValueError(f'The attribute list has duplicated values.')
        else:
            found_attributes[attribute] = True

def bar_to_encoding(bar: int) -> Tuple[int, int]:
    """
    Function converts bar indexes to model encoding value.
    Bar encodings start from 4 - 259 representing <0-0> -> <0-255>
    therefore are just shifted by a value of 4. Seperate function
    in case shift needs to change.
    @author: Stephen Krol

    Args:
        bar_idx (int): bar value
    Returns:
        int: encoding bar value
    """

    return bar + 4

def get_bar_idxs(encoding: torch.tensor, bar_start: int, bar_end: int):
    '''
    Function takes an encoding vector and finds the indices that correspond to the bars in
    a slice range.
    @author: sjkrol

    Args:
        encoding (tensor): the input song encoded into octuple encoding.
        label_dict (dict): label dictionary for MusicBert model
        bar_start (int): starting bar inclusive
        bar_end (int): ending bar inclusive
    Returns:
        int: starting index (inclusive) of bar in encoding tensor
        int: ending index (inclusive) of bar in encoding tensor
    '''

    assert bar_start >= 0 and bar_start < 255, "Starting bar must be between 0 and 255"
    assert bar_end >= 0 and bar_end < 255, "End bar must be between 0 and 255"
    assert bar_start <= bar_end, "Start of bar must be less than end of bar"

    max_track_bar = int(encoding[-16]) # final note bar ocutple

    # bar encodings start from 4 - 259 representing <0-0> -> <0-255>
    bar_start_encoding, bar_end_encoding = bar_to_encoding(bar_start), bar_to_encoding(bar_end)

    assert bar_start_encoding <= max_track_bar, f"{bar_start} out of range."
    assert bar_end_encoding <= max_track_bar, f"{bar_end} out of range."

    start_idx = None
    end_idx = None

    # find first instance of bar start encoding
    for i in range(0, len(encoding), 8):
        encoded_event = encoding[i]
        if encoded_event.item() == bar_start_encoding:
            start_idx = i
            break
        elif encoded_event.item() < bar_start_encoding:
            start_idx = i # next best option, last octuple of last bar
        else:
            break # exit if gone beyond start bar
    
    # find first instance of bar end encocding
    for i in reversed(range(0, len(encoding)-8, 8)):
        encoded_event = encoding[i]

        if encoded_event.item() == bar_end_encoding:
            end_idx = i + 7 # include all info from octuple
            break
        elif encoded_event.item() > bar_end_encoding:
            end_idx = i - 1 # set to previous octuple end, will only return if end bar is empty
        else:
            break # exit if beyond end bar


    return start_idx, end_idx

def get_bar_octuples(encoding: torch.tensor, bars=None, bar_separation=False):
    '''
    Function takes an encoding tensor and a list of bars that can be masked. Each bar is a tuple
    containing either one or two bar numbers. If the tuple contains a single number then only that
    bar will be masked. If the tuple contains two numbers, every bar inbetween those numbers will
    be masked.
    @author: sjkrol

    Args:
        encoding (tensor): the input song encoded into octuple encoding.
        bars (list): list of bars that can be masked.
        bar_separation (bool): if true, keep octuples of each bar in a separate list
    Returns:
        list: list of octuples that can be masked. 
    '''

    assert bar_order_check(bars) is True, "Provided bars are not in ascending order and/or there are overlapping bars."
    maskable_octuples = []

    for bar in bars:
        # single bar
        if type(bar) is int:
            start_idx, end_idx = get_bar_idxs(encoding, bar, bar) 
        # bar range
        else:
            start_idx, end_idx = get_bar_idxs(encoding, bar[0], bar[1])
            
        octuples = [i for i in range(int(start_idx / 8), int((end_idx+1) / 8))]

        if bar_separation:
            maskable_octuples.append(octuples)
        else:
            maskable_octuples.extend(octuples)

    return maskable_octuples

def filter_octuples_pitch(encoding: torch.tensor, 
                          min_pitch:int=None,
                          max_pitch:int=None):
    """
    Function filters out octuples where the pitch is lower
    than a specified pitch. Used to ensure only melodic or
    harmonic varition.
    @author: Stephen Krol

    Args:
        encoding (torch.tensor): unflitered encoding vector.
        min_pitch (int): any note below this pitch will be
            filtered out.
    Returns:
        list: filtered octuples containing only notes above a certain
            pitch value.
    """

    filtered_octuples = []

    if min_pitch is None:
        min_pitch = 0
    
    if max_pitch is None:
        max_pitch = np.inf

    for octuple in range(1, int( len(encoding) / 8 ) - 2):

        pitch = encoding[octuple*8 + 3]
        if pitch >= min_pitch and pitch <= max_pitch:
            filtered_octuples.append(octuple)

    return filtered_octuples 

def filter_octuples(encoding: torch.tensor,
                    min_pitch: int,
                    max_pitch: int,
                    bars: list):
    """
    Function filters through valid octuples to vary based
    on user input parameters of min/max pitch and bars.
    # TODO: improve this, this feels ugly and wrong
    @author: Stephen Krol

    Args:
        encoding (torch.tensor): encoded midi sequence for
            MusicBert.
        min_pitch (int): minimum pitched note that can be
            varied.
        max_pitch (int): maximum pitched note that can be 
            varied.
        bars (list): a list of bars that can be masked, contains 
            either ints representing single bars or tuples containing
            a bar range.
    Returns:
        list-like: a list of valid octuples that can be varied.
    """
    pitch_octuples = []
    bar_octuples = []

    if min_pitch is not None or max_pitch is not None:
        pitch_octuples = filter_octuples_pitch(encoding, min_pitch, max_pitch) # TODO: might be fitering out new notes

    if bars:
        bar_octuples = get_bar_octuples(encoding=encoding, bars=bars)
    
    # 
    if pitch_octuples or bar_octuples:
        # if only bars:
        if not pitch_octuples:
            print("Only Bars")
            return bar_octuples
        # if only pitch
        elif not bar_octuples:
            print("only Pitch")
            return pitch_octuples
        else:
            print("Both")
            # this is definitely not the best way to do this
            # running out of time though :(
            return np.intersect1d(bar_octuples, pitch_octuples)
    else:
        return range(1, int( len(encoding) / 8 ) - 2)

def filter_key(probs: torch.tensor,
               key:str,
               chromatic_notes: np.ndarray,
               key_notes: np.ndarray,
               beta=int) -> torch.tensor:
    """
    Function filters key probabilities based off user input.
    If beta=1/P, where P is the culmative probability of chromatic notes,
    then only chromatic notes will be played.
    @author: Stephen Krol

    Args:
        Probs (torch.tensor): Probability of each pitch from model.
        key (str): key of the song.
        beta (int): 0 <= beta <= 1, controls change in proabability for
            chromatic notes.
    Returns:
        torch.tensor: updated probability distribution.
    """

    chromatic_probs = probs[chromatic_notes]
    key_probs = probs[key_notes]

    chromatic_cumulative = torch.sum(chromatic_probs)
    if chromatic_cumulative == 0:
        beta=0
    else:
        # rescale to be between 0 - 1/P
        beta = beta*(1/torch.sum(chromatic_probs))
    

    alpha = (1 - beta*torch.sum(chromatic_probs)) / torch.sum(key_probs)

    # rescale probabilities
    probs[chromatic_notes] = probs[chromatic_notes]*beta
    probs[key_notes] = probs[key_notes]*alpha

    return probs

def controlled_masking(encoding: torch.tensor,
                       attributes: list, 
                       note_percentage_random_mask: int, 
                       mask_idx: int, 
                       min_pitch:int=None,
                       max_pitch:int=None,
                       bars=None,
                       new_notes=None):
    '''
    Function allows users to control which attributes are masked, which bars this masking is applied to and how many
    of the notes are masked.
    @author: midiformer
    @modified: sjkrol

    Args:
        encoding (tensor): an encoding tensor for the MusicBERT
            model.
        attributes (list): a list containing which attributes to mask
            0: bar | 1: position | 2: instrument | 3: pitch | 4: duration | 5: velocity | 6: time signature | 7: tempo 
        note_percentage_random_mask (int): ranging from 0-100, specifies the percentage of notes to mask
        mask_idx (int): encoded value for the <mask> token
        min_pitch (int): any note below this pitch will be filtered out.
        max_pitch (int): any note above this pitch will be filtered out.
        bars (list): a list of bars that can be masked, contains either ints representing single bars or tuples containing
            a bar range.
        new_notes (np.ndarray): array containing octuple positions that contain new notes.
    Returns:
        (tensor): encoding tensor for MusicBert with masked octuples.
    '''

    # create copy in case of multiple variations
    encoding = encoding.clone()

    # ensure attributes array is of the correct form
    attributes_check(attributes)

    # if user specifies bars only mask octuples from those bars
    octuples = filter_octuples(encoding=encoding,
                               min_pitch=min_pitch,
                               max_pitch=max_pitch,
                               bars=bars)

    # if new notes have beeen added, prevent masking of those new notes
    if new_notes is not None:
        octuples = np.setdiff1d(octuples, new_notes)

    # randomly select octuples to mask
    masked_octs = np.random.choice(a=octuples, 
                                    size=int(len(octuples)*(note_percentage_random_mask/100)),
                                    replace=False)
    
    for masked_oct in masked_octs:
        encoding.index_fill_(0, torch.tensor( masked_oct * 8 + attributes , dtype=torch.int64), mask_idx)
    
    return encoding

def bar_level_masking(encoding: torch.tensor,
                      attributes: list,
                      bars: list,
                      mask_idx: int):
    '''
    Takes an encoding and masks specified attributes at a bar-level in
    specified bars.
    @author: sjkrol

    Args:
        encoding (torch.tensor): an encoding tensor for the MusicBERT model.
        attributes (list): a list containing which attributes to mask
            0: bar | 1: position | 2: instrument | 3: pitch | 4: duration | 5: velocity | 6: time signature | 7: tempo 
        bars (list): a list of bars that can be masked, contains either ints representing single bars or tuples containing
            a bar range.
        mask_idx (int): encoded value for the <mask> token
    
    Returns:
        (tensor): encoding tensor for MusicBert with bar level masked octuples.
    '''

    encoding = encoding.clone()

    octuples = get_bar_octuples(encoding, bars, bar_separation=False)

    # TODO: work on bar level, for now combine

    for masked_oct in octuples:
        encoding.index_fill_(0, torch.tensor( np.int32(masked_oct) * 8 + attributes ), mask_idx)

    return encoding

def get_invalid_bars(bars: list) -> list:
    """
    Function takes valid prediction bars from user and 
    returns a list of bars that should not be predicted.
    @author: Stephen Krol

    Args:
        bars (list): list containing valid bar ranges.
    Returns:
        list: a list of bars that can be masked from
            prediction. 
    """
    invalid_bars = []
    current_bar = 0
    curent_bar_start = bars[current_bar][0]
    current_bar_end = bars[current_bar][1]
    max_bars = 255
    i = 0
    # iterate over range of total bars
    while i < max_bars:
        if i < curent_bar_start: # current bar less than the valid range
            invalid_bars.append(bar_to_encoding(i))
            i += 1
        else:
            i = current_bar_end + 1 # set to after current range
            if current_bar < len(bars)-1: 
                current_bar+=1
                curent_bar_start = bars[current_bar][0]
                current_bar_end = bars[current_bar][1]
            else:
                curent_bar_start = np.inf
    
    return invalid_bars

def vanilla_prediction(roberta_base: MusicBERTModel, 
                       encoding: torch.tensor, 
                       mask_idx: int,
                       label_dict: dict,
                       reversed_dict: dict,
                       temperature_dict: dict,
                       multinomial_sample: bool,
                       min_pitch:int=None,
                       max_pitch:int=None,
                       bars:list=None,
                       key:str=None,
                       beta:int=0,
                       custom_progress_bar=None,
                       playback_button=None):
    '''
    Predicts each masked token sequentially.
    (1) Find first mask token and predict.
    (2) Add token to sequence.
    (3) Repeat
    @author: midiformers
    @modified: sjkrol

    Args:
        roberta_base (MusicBERTModel): pre-trained music bert model.
        encoding (torch.tensor): an encoding tensor of music for the music bert model.
        mask_idx (int): token representing masked index.
        label_dict (dict): dictionary for str to token mapping.
        reversed_dict (dict): dictionary for token to str mapping.
        temperature_dict (dict): dictionary containing temperature values for each
            attribute.
        multinomial_sample (bool): if True, samples attribute from a multinomial distribution
            regardless of temperature value.
        min_pitch (int): any note BELOW this pitch will be filtered out of variation process.
        max_pitch (int): any note ABOVE this pitch will be filtered out of variation process.
        bars (list): range of bars to predict in.
        key (str): key of song.
        beta (int): weighting term, controls likelihood of chromatic note prediction.
        custom_progress_bar (customtkinter.CTkProgressBar): if not None, update this progress
            bar.
        playback_button (customtkinter.CTkButton): if not None, update this button on completion.
    Returns:
        torch.tensor: the encoding tensor of music for music bert with all masked tokens
            predicted.
    '''
   
    masked_idxs = [i for i, x in enumerate(encoding.tolist()) if x==mask_idx]
    encoding_dtype = encoding.dtype

    # prepare for pitch filtering
    pitch_idxs = list(range(*pitch_range(label_dict)))

    # PITCH FILTERING
    # TODO: this is ugly, there will be a better solution that does not use None
    if min_pitch is not None:
        min_pitch_idx = min_pitch - MIN_ENCODED_PITCH
    else:
        min_pitch_idx = 0
    if max_pitch is not None:
        max_pitch_idx = max_pitch - MIN_ENCODED_PITCH
    else:
        max_pitch_idx = len(pitch_idxs)
    filtered_pitches_idx = pitch_idxs[:min_pitch_idx] + pitch_idxs[max_pitch_idx+1:]

    # KEY FILTERING
    print(key)
    if key is not None:
        key_notes, chromatic_notes = get_key_notes(key)

    # BAR FILTERING
    if bars is None: invalid_bars = None
    else: invalid_bars = get_invalid_bars(bars)

    for i, masked_idx in enumerate(tqdm(masked_idxs)):
        
        # update progress bar
        if custom_progress_bar is not None:
            custom_progress_bar.set(i / (len(masked_idxs)-1))

        input = encoding.unsqueeze(0)
        prev_idx = encoding[masked_idx-1]
        # print(reversed_dict[prev_idx.item()])
        with torch.no_grad():
            features, _ = roberta_base.model.extract_features(input)
            logits = features[0, masked_idx]

            # retrieve temperature based off previous attirbute
            # octuple encoding is cyclical
            temperature = switch_temperature(prev_idx, reversed_dict, temperature_dict)
            if temperature != 1: logits = logits / temperature

            logits = filter_invalid_indexes(logits, 
                                            prev_idx, 
                                            label_dict, 
                                            reversed_dict,
                                            filtered_pitches_idx=filtered_pitches_idx,
                                            invalid_bars=invalid_bars)
            # TODO: investigate top_k performance
            probs = torch.softmax(logits, dim=-1)
            # probs = top_k_top_p(probs, top_k=5, filter_value=0)
            str_encoding = reversed_dict[prev_idx.item()]
            if str_encoding[1] == '2' and key is not None:
                probs = filter_key(probs, key, chromatic_notes, key_notes, beta)

            if probs.dim() == 1:
                probs = probs.unsqueeze(0)

            # TODO: more temperature and repeat count
            # TODO: improve temperature
            # TODO: what is repeat count
            if temperature != 1 or multinomial_sample:
                top_preds = torch.multinomial(probs, 1)
                top_preds = top_preds.reshape(-1)
                # print(reversed_dict[top_preds.item()])
            else:
                top_preds = torch.argmax(probs, dim=1)
            
            encoding[masked_idx] = top_preds.type(encoding_dtype)
    
    # update GUI button
    if playback_button is not None:
        playback_button.configure(state='normal', fg_color='#1F6AA5')
    
    return encoding

def get_total_bars(encoding: torch.tensor, bars:list=None):
    """
    Function grabs the maximum bar number from
    the encoding. Assumes the last octuple in
    encoding is the last note in the track.
    @author: Stephen Krol

    Args:
        encoding (torch.tensor): encoding of midi file.
    Returns:
        int: max bar in the track.
        int: total number of notes available in bars
    """

    if bars is None:
        total_bars = int(encoding[-16] - 4) + 1 # final note bar ocutple
        return [i for i in range(total_bars)] , (len(encoding) / 8) - 2
    else:
        # add up total bars from provided ranges
        total_bars = []
        total_notes = 0
        for bar_range in bars:
            for i in range(bar_range[0], bar_range[1] + 1):
                total_bars.append(i)
            start_idx, end_idx = get_bar_idxs(encoding, bar_range[0], bar_range[1])
            total_notes += ((end_idx+1) - start_idx)/8


        return total_bars, total_notes

# TODO: ensure perctange of new notes based off available octuples in bar range
# TODO: if no bar range then spread evenly
def add_notes(encoding: torch.tensor, 
              new_notes_percentage: int, 
              mask_idx: int,
              bars:list=None) -> Tuple[torch.tensor, np.ndarray]:
    '''
    Adds new notes to the song. New notes are added as masked tokens which are then
    predicted using MusicBert.
    @author: sjkrol

    Args:
        encoding (torch.tensor): an encoding vector for a musical phrase for MusicBert.
        new_notes_percentage (int): number of new notes to add as a percentae of the
            number of notes in the encoding vector.
        mask_idx (int): masking token.
        bars (list): a list of bars that be varied.
    Returns:
        torch.tensor: an encoding vector with added masked notes.
        np.ndarray: new notes octuple number.
    '''

    assert len(encoding) % 8 == 0, "encoding input must be divisible by 8"

    # calculate new notes
    n_octuples = int(len(encoding) / 8)
    n_new_octuples = int(n_octuples * (new_notes_percentage/100))

    # calculate number of bars to add notes to
    if bars is None: bars = get_total_bars(encoding)[0] # currently adding total notes, can be used to control %
    else: bars = get_total_bars(encoding, bars)[0]

    n_bars = len(bars)
    max_bar = int(encoding[-16] - 4) + 1

    # caluclate new notes per bar
    n_new_per_bar = n_new_octuples // n_bars
    if n_new_per_bar == 0: n_new_per_bar=1

    print(f'Number of Octuples: {n_octuples}')
    print(f'new notes per bar: {n_new_per_bar}')
    print(f'number of bars: {n_bars}')

    # retrieve instrument, assumes all insturments are the same
    instrument = encoding[(2 * 8) + 2]

    # create new encoding vector
    new_encoding = torch.zeros((n_octuples+(n_new_per_bar*n_bars)) * 8, dtype=encoding.dtype)

    masked_octs = []
    # iterate through each bar and randomly select octuples
    # TODO: fix empty bars, currently need to see if you can work with new solution
    # is the start_idx already covered? probably, empty bars will have new notes in them
    # so need to find their indexes
    for i, bar in enumerate(bars):
        start_idx, end_idx = get_bar_idxs(encoding, bar, bar)
        shift = (i*n_new_per_bar*8) # compensate for added new notes
        start_octuple = int((shift+start_idx)/8)
        end_octuple = int((shift+end_idx+1)/8) #+ n_new_per_bar

        if bar != max_bar-1:
            masked_oct = np.random.choice( a = range(start_octuple, end_octuple+n_new_per_bar) , \
                        size = n_new_per_bar, \
                        replace = False)
        else:
            masked_oct = np.random.choice( a = range(start_octuple, end_octuple+n_new_per_bar-1) , \
                        size = n_new_per_bar, \
                        replace = False)
        masked_octs.append(masked_oct)
        
        for oct in masked_oct:
            new_encoding[oct*8 : (oct+1)*8] = mask_idx

        new_encoding[masked_oct*8] = bar_to_encoding(bar)
        new_encoding[(masked_oct*8)+2] = instrument

    masked_octs = np.concatenate(masked_octs)
    print(len(encoding))
    # fill new encoding vector
    i = 0
    for j in range(len(new_encoding)):
        if new_encoding[j] !=0:
            continue
        else:
            new_encoding[j] = encoding[i].type(new_encoding.dtype)
            i += 1
            if i == len(encoding):
                new_encoding = new_encoding[0:j+1]
                print(new_encoding)
                break

    if len(new_encoding) % 8 != 0:
        raise ValueError("Encoding not divisible by 8")
    

    return new_encoding, masked_octs

def create_attribute_array(attributes: list) -> list:
    """
    Function takes a list containing the attributes
    to vary and generates the attribute array. This array
    is used in the masking function and the values of 
    each attibute are:
    0: bar | 1: position | 2: instrument | 3: pitch | 4: duration | 
    5: velocity | 6: time signature | 7: tempo 
    @author: Stephen Krol

    Args:
        attributes (list): list containing which attributes
            are being varied.
    Returns:
        list: array containing the attribute indexes to change based
            off the octuple encoding.
    """
    
    return [ATTRIBUTE_INDEXES[attribute] for attribute in attributes]

def create_temperature_dict(temperatures: dict) -> dict:
    """
    Function creates function dictionary from inputed
    temperature values.
    @author: Stephen Krol

    Args:
        temperatures (dict): key - attribute name as a str
            value - temperature value.
    Returnns:
        (dict): key - attribute as index value
            value - temperature value.
    """

    temperature_dict = {
        0 : 1.0,
        1 : 1.0,
        2 : 1.0,
        3 : 1.0,
        4 : 1.0,
        5 : 1.0,
        6 : 1.0,
        7 : 1.0,
    }

    for attribute, temperature in temperatures.items():
        temperature_dict[ATTRIBUTE_INDEXES[attribute]] = temperature

    return temperature_dict

def generate_variations(filename: str, 
                        n_var: int, 
                        roberta_base: MusicBERTModel,
                        label_dict: dict,
                        reversed_dict: dict,
                        new_notes: bool,
                        new_notes_percentage: int,
                        variation_percentage: int,
                        attributes: list, 
                        temperature_dict: dict, 
                        min_pitch:int=None,
                        max_pitch:int=None,
                        key:str=None,
                        beta:int=0,
                        bars=None,
                        bar_level=False,
                        multinomial_sample=False,
                        custom_progress_bars=None,
                        playback_buttons=None,
                        GUI:bool=True):
    '''
    Takes a midi filepath and generates n variations using the MusicBert model over specified
    attributes and controllable temperature.
    @author: sjkrol

    Args:
        filename (str): filepath to midi file
        n_var (int): number of variations to generate
        roberta_base (MusicBERTModel): pretrained music bert model
        label_dict (dict): dictionary to convert string to token
        reversed_dict (dict): dictionary to convert token to string
        add_notes (bool): if True, new notes will be added to the variation
        new_notes_percentage (int): how many new notes should be added to the piece as a percentage of notes
            in the original piece.
        variation_percentage (int): how much of the piece should be varied
        attributes (list): list containing attributes to vary
        0: bar | 1: position | 2: instrument | 3: pitch | 4: duration | 5: velocity | 6: time signature | 7: tempo
        temperature_dict (dict): dictionary containing temperature values for each attribute
        min_pitch (int): any note BELOW this pitch will be filtered out of variation process.
        max_pitch (int): any note ABOVE this pitch will be filtered out of variation process.
        key (str): key of song.
        beta (int): weighting term, controls likelihood of chromatic note prediction.
        bars (list or None): if None vary over all bars, if a list only vary bars in the list
        bar_level (bool): if True, mask all elements in a bar
        multinomial_sample (bool): if True, samples attribute from a multinomial distribution
            regardless of temperature value.
        custom_progress_bar (list): if not None, update these progress
            bars. FOR USE WITH CUSTOMTKINTER GUI
        playback_buttons (list): if not none, update the status of these buttons.
            FOR USE WITH CUSTOMTKINTER GUI
        GUI (bool): if True, function is being called through the GUI.
    Returns:
        list: a list containing n variations
    '''

    # set MusicBert to evaluation mode
    roberta_base.eval()

    # identify mask toke
    mask_idx = label_dict.index('<mask>')

    # TODO: add assertion that all instruments are the same
    # encode midi file
    encoding = encode_midi_for_musicbert(filename, label_dict)

    if GUI:
        # create attribute indexed array
        attributes = create_attribute_array(attributes)

        # create complete temperture dictionary
        temperature_dict = create_temperature_dict(temperature_dict)

    # TODO: this is horrible, find a way of not assigning none values to min pitch and max pitch, shouldnt be hard
    # check max and min pitch values
    if min_pitch is not None:
        assert min_pitch >= MIN_ENCODED_PITCH, f"Invalid min pitch value, should range [{MIN_ENCODED_PITCH} - {MAX_ENCODED_PITCH}]"
    
    if max_pitch is not None:
        assert max_pitch <= MAX_ENCODED_PITCH, f"Invalid max pitch value, should range [{MIN_ENCODED_PITCH} - {MAX_ENCODED_PITCH}]"

    if max_pitch is not None and min_pitch is not None:
        assert min_pitch <= max_pitch, f"min pitch must be less that max pitch"

    # create variations list
    variations = []

    # generate n variaitons
    for i in range(n_var):

        # add notes
        if new_notes is True:
            encoding, new_notes = add_notes(encoding, new_notes_percentage, mask_idx, bars=bars)

        if bar_level:
            masked_encoding = bar_level_masking(encoding, attributes=attributes, bars=bars, mask_idx=mask_idx)
        else:
            masked_encoding = controlled_masking(encoding, 
                                                attributes, 
                                                note_percentage_random_mask=variation_percentage, 
                                                min_pitch=min_pitch,
                                                max_pitch=max_pitch,
                                                mask_idx=mask_idx,
                                                bars=bars,
                                                new_notes=new_notes)
        if custom_progress_bars is not None:
            custom_progress_bar = custom_progress_bars[i]
        else:
            custom_progress_bar=None

        if playback_buttons is not None:
            playback_button = playback_buttons[i]
        else:
            playback_button = None

        # run prediction
        pred_encoding = vanilla_prediction(roberta_base, 
                                           masked_encoding, 
                                           mask_idx, 
                                           label_dict, 
                                           reversed_dict, 
                                           temperature_dict,
                                           multinomial_sample,
                                           min_pitch,
                                           max_pitch,
                                           bars,
                                           key,
                                           beta,
                                           custom_progress_bar,
                                           playback_button)

        variations.append(pred_encoding)

    return variations

def write_variation(variation, filepath_prefix: str, reversed_dict: dict):
    '''
    Writes all variations to specified directory.
    @author: sjkrol

    Args:
        variation (torch.tensor): encoded midi output for variation.
        filepath_prefix (str): the filepath and filename prefix that the variations will
            be saved to.
        reversed_dict (dict): dictionary for converting tokens to str.
    Returns:
        None
    '''

    encoding_str = decode_w_label_dict(reversed_dict, variation)
    octuple_encoding = str_to_encoding(encoding_str)
    midi_obj = encoding_to_MIDI(encoding=octuple_encoding)
    midi_obj.dump(f'{filepath_prefix}')


if __name__ == "__main__":


    filename = '../Pitch Test.mid'

    if not os.path.exists('/home/sjkro1/muzic/musicbert/input0/dict.txt'):
       gen_dictionary("input0/dict.txt")
    
    if not os.path.exists('/home/sjkro1/muzic/musicbert/label/dict.txt'):
       gen_dictionary("label/dict.txt")

    roberta_base = MusicBERTModel.from_pretrained('.', 
        checkpoint_file='../../RhapsodyRefiner/static/Weights/checkpoint_last_musicbert_base_w_genre_head.pt'
    )

    roberta_base.eval()

    # extract label dict and create reverse label dict
    label_dict = roberta_base.task.label_dictionary
    reversed_dict = reverse_label_dict(label_dict)

    # set temperature
    temp = 1 

    temp_bar = 1
    temp_pos = 2
    temp_ins = 1
    temp_pitch = 2
    temp_dur = 1
    temp_vel = 1
    temp_sig = 1
    temp_tempo = 1

    # create temperature dict
    temperature_dict = {
        0 : temp_bar,
        1 : temp_pos,
        2 : temp_ins,
        3 : temp_pitch,
        4 : temp_dur,
        5 : temp_vel,
        6 : temp_sig,
        7 : temp_tempo,
    }


    # 0: bar | 1: position | 2: instrument | 3: pitch | 4: duration | 5: velocity | 6: time signature | 7: tempo 
    attributes = [3]
    bars = [(2,2)]
    # bars = None

    variations = generate_variations(filename=filename, 
                                     n_var=1, 
                                     roberta_base=roberta_base,
                                     label_dict=label_dict, 
                                     reversed_dict=reversed_dict, 
                                     new_notes=True, 
                                     new_notes_percentage=10, 
                                     variation_percentage=50, 
                                     multinomial_sample=True,
                                     attributes=attributes, 
                                     key='C Major',
                                     temperature_dict=temperature_dict, 
                                     bars=bars, 
                                     bar_level=False,
                                     GUI=False)
    
    write_variation(variations[0], "test.mid", reversed_dict)

