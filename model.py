import tensorflow as tf
import os
import pickle

from tokenizer.encoder_decoder import build_sequence_encode_decode_dicts


def build_params(input_data=None,
                 output_data=None,
                 params_path='test_params',
                 max_lenghts=(5, 5)):
    if output_data is None:
        output_data = []
    if input_data is None:
        input_data = []
    if os.path.exists(params_path):
        print('Loading the params file')
        params = pickle.load(open(params_path, 'rb'))
        return params

    print('Creating params file')
    input_encoding, input_decoding, input_dict_size = build_sequence_encode_decode_dicts(input_data)
    output_encoding, output_decoding, output_dict_size = build_sequence_encode_decode_dicts(output_data)
    params = {
        'input_encoding': input_encoding,
        'input_decoding': input_decoding,
        'input_dict_size': input_dict_size,
        'output_encoding': output_encoding,
        'output_decoding': output_decoding,
        'output_dict_size': output_dict_size,
        'max_input_length': max_lenghts[0],
        'max_output_length': max_lenghts[1]
    }

    pickle.dump(params, open(params_path, 'wb'))
    return params


def build_model(params_path='test/params',
                enc_lstm_units=128,
                unroll=True,
                use_gru=False,
                optimizer='adam',
                display_summary=True):
    params = build_params(params_path=params_path)

    input_encoding = params['input_encoding']
    input_decoding = params['input_decoding']
    input_dict_size = params['input_dict_size']
    output_encoding = params['output_encoding']
    output_decoding = params['output_decoding']
    output_dict_size = params['output_dict_size']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    if display_summary:
        print('Input encoding', input_encoding)
        print('Input decoding', input_decoding)
        print('Output encoding', output_encoding)
        print('Output decoding', output_decoding)

    # We need to define the max input lengths and max output lengths before training the model.
    # We pad the inputs and outputs to these max lengths
    encoder_input = tf.keras.layers.Input(shape=(max_input_length,))
    decoder_input = tf.keras.layers.Input(shape=(max_output_length,))

    # Need to make the number of hidden units configurable
    encoder = tf.keras.layers.Embedding(input_dict_size, enc_lstm_units, input_length=max_input_length, mask_zero=True)(
        encoder_input)
    # using concat merge mode since in my experiments it gave the best results same with unroll
    if not use_gru:
        encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(enc_lstm_units, return_sequences=True, return_state=True, unroll=unroll),
            merge_mode='concat')(encoder)
        encoder_outs, forward_h, forward_c, backward_h, backward_c = encoder
        encoder_h = tf.keras.layers.concatenate([forward_h, backward_h])
        encoder_c = tf.keras.layers.concatenate([forward_c, backward_c])

    else:
        encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(enc_lstm_units, return_sequences=True, return_state=True, unroll=unroll),
            merge_mode='concat')(encoder)
        encoder_outs, forward_h, backward_h = encoder
        encoder_h = tf.keras.layers.concatenate([forward_h, backward_h])

    # using 2* enc_lstm_units because we are using concat merge mode
    # cannot use bidirectionals lstm for decoding (obviously!)

    decoder = tf.keras.layers.Embedding(output_dict_size, 2 * enc_lstm_units, input_length=max_output_length,
                                        mask_zero=True)(
        decoder_input)

    if not use_gru:
        decoder = tf.keras.layers.LSTM(2 * enc_lstm_units, return_sequences=True, unroll=unroll)(decoder,
                                                                                                 initial_state=[
                                                                                                     encoder_h,
                                                                                                     encoder_c])
    else:
        decoder = tf.keras.layers.GRU(2 * enc_lstm_units, return_sequences=True, unroll=unroll)(decoder,
                                                                                                initial_state=encoder_h)

    # long attention
    attention = tf.keras.layers.dot([decoder, encoder_outs], axes=[2, 2])
    attention = tf.keras.layers.Activation('softmax', name='attention')(attention)

    context = tf.keras.layers.dot([attention, encoder_outs], axes=[2, 1])

    decoder_combined_context = tf.keras.layers.concatenate([context, decoder])

    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(enc_lstm_units, activation="tanh"))(
        decoder_combined_context)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_dict_size, activation="softmax"))(output)

    model = tf.keras.models.Model(inputs=[encoder_input, decoder_input], outputs=[output])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if display_summary:
        model.summary()

    return model, params
