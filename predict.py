import numpy as np

from tokenizer.encoder_decoder import encode_sequences, decode_sequence


def generate(texts,
             input_encoding_dict,
             model,
             max_input_length,
             max_output_length,
             char_start_encoding,
             beam_size,
             max_beams,
             min_cut_off_len,
             cut_off_ratio):
    if not isinstance(texts, list):
        texts = [texts]

    min_cut_off_len = max(min_cut_off_len, cut_off_ratio * len(max(texts, key=len)))
    min_cut_off_len = min(min_cut_off_len, max_output_length)

    all_completed_beams = {i: [] for i in range(len(texts))}
    all_running_beams = {}
    for i, text in enumerate(texts):
        all_running_beams[i] = [[np.zeros(shape=(len(text), max_output_length)), [1]]]
        all_running_beams[i][0][0][:, 0] = char_start_encoding

    while len(all_running_beams) != 0:
        for i in all_running_beams:
            all_running_beams[i] = sorted(all_running_beams[i], key=lambda tup: np.prod(tup[1]), reverse=True)
            all_running_beams[i] = all_running_beams[i][:max_beams]

        in_out_map = {}
        batch_encoder_input = []
        batch_decoder_input = []
        t_c = 0
        for text_i in all_running_beams:
            if text_i not in in_out_map:
                in_out_map[text_i] = []
            for running_beam in all_running_beams[text_i]:
                in_out_map[text_i].append(t_c)
                t_c += 1
                batch_encoder_input.append(texts[text_i])
                batch_decoder_input.append(running_beam[0][0])

        batch_encoder_input = encode_sequences(input_encoding_dict, batch_encoder_input, max_input_length)
        batch_decoder_input = np.asarray(batch_decoder_input)
        batch_predictions = model.predict([batch_encoder_input, batch_decoder_input])

        t_c = 0
        for text_i, t_cs in in_out_map.items():
            temp_running_beams = []
            for running_beam, probs in all_running_beams[text_i]:
                if len(probs) >= min_cut_off_len:
                    all_completed_beams[text_i].append([running_beam[:, 1:], probs])
                else:
                    prediction = batch_predictions[t_c]
                    sorted_args = prediction.argsort()
                    sorted_probs = np.sort(prediction)

                    for i in range(1, beam_size + 1):
                        temp_running_beam = np.copy(running_beam)
                        i = -1 * i
                        ith_arg = sorted_args[:, i][len(probs)]
                        ith_prob = sorted_probs[:, i][len(probs)]

                        temp_running_beam[:, len(probs)] = ith_arg
                        temp_running_beams.append([temp_running_beam, probs + [ith_prob]])

                t_c += 1

            all_running_beams[text_i] = [b for b in temp_running_beams]

        to_del = []
        for i, v in all_running_beams.items():
            if not v:
                to_del.append(i)

        for i in to_del:
            del all_running_beams[i]

    return all_completed_beams


def infer(texts, model, params, beam_size=3, max_beams=3, min_cut_off_len=10, cut_off_ratio=1.5, char_start_encoding=1):
    if not isinstance(texts, list):
        texts = [texts]

    input_encoding_dict = params['input_encoding']
    output_decoding_dict = params['output_decoding']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    all_decoder_outputs = generate(texts,
                                   input_encoding_dict,
                                   model,
                                   max_input_length,
                                   max_output_length,
                                   char_start_encoding,
                                   beam_size,
                                   max_beams,
                                   min_cut_off_len,
                                   cut_off_ratio)
    outputs = []

    for i, decoder_outputs in all_decoder_outputs.items():
        outputs.append([])
        for decoder_output, probs in decoder_outputs:
            outputs[-1].append(
                {'sequence': decode_sequence(output_decoding_dict, decoder_output[0]), 'prob': np.prod(probs)})

    return outputs


def generate_greedy(texts,
                    input_encoding_dict,
                    model,
                    max_input_length,
                    max_output_length,
                    char_start_encoding,
                    char_padding_encoding):
    if not isinstance(texts, list):
        texts = [texts]

    encoder_input = encode_sequences(input_encoding_dict, texts, max_input_length)
    decoder_input = np.zeros(shape=(len(encoder_input), max_output_length))
    decoder_input[:, 0] = char_start_encoding
    for i in range(1, max_output_length):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:, i] = output[:, i]

        if np.all(decoder_input[:, i] == char_padding_encoding):
            return decoder_input[:, 1:]

    return decoder_input[:, 1:]


def infer_greedy(texts, model, params, char_start_encoding=1, char_padding_encoding=0):
    return_string = False
    if not isinstance(texts, list):
        return_string = True
        texts = [texts]

    input_encoding_dict = params['input_encoding']
    output_decoding_dict = params['output_decoding']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    decoder_output = generate_greedy(texts,
                                     input_encoding_dict,
                                     model,
                                     max_input_length,
                                     max_output_length,
                                     char_start_encoding,
                                     char_padding_encoding)
    if return_string:
        return decode_sequence(output_decoding_dict, decoder_output[0])

    return [decode_sequence(output_decoding_dict, i) for i in decoder_output]