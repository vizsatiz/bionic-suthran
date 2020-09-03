import numpy as np


def build_sequence_encode_decode_dicts(input_data):
    encoding_dict = {}
    decoding_dict = {}
    for line in input_data:
        for char in line:
            if char not in encoding_dict:
                # Using 2 + because our sequence start encoding is 1 and padding encoding is 0
                encoding_dict[char] = 2 + len(encoding_dict)
                decoding_dict[2 + len(decoding_dict)] = char
    return encoding_dict, decoding_dict, len(encoding_dict) + 2


def encode_sequences(encoding_dict, sequences, max_length):
    encoded_data = np.zeros(shape=(len(sequences), max_length))
    for i in range(len(sequences)):
        for j in range(min(len(sequences[i]), max_length)):
            encoded_data[i][j] = encoding_dict[sequences[i][j]]
    return encoded_data


def decode_sequence(decoding_dict, sequence):
    text = ''
    for i in sequence:
        if i == 0:
            break
        text += decoding_dict[i]
    return text
