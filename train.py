import numpy as np
import tensorflow as tf

from tokenizer.encoder_decoder import encode_sequences

np.random.seed(6788)

char_start_encoding = 1
char_padding_encoding = 0


def convert_training_data(input_data, output_data, params):
    input_encoding = params['input_encoding']
    output_encoding = params['output_encoding']
    output_dict_size = params['output_dict_size']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    encoded_training_input = encode_sequences(input_encoding, input_data, max_input_length)
    encoded_training_output = encode_sequences(output_encoding, output_data, max_output_length)
    training_encoder_input = encoded_training_input
    training_decoder_input = np.zeros_like(encoded_training_output)
    training_decoder_input[:, 1:] = encoded_training_output[:, :-1]
    training_decoder_input[:, 0] = char_start_encoding
    training_decoder_output = np.eye(output_dict_size)[encoded_training_output.astype('int')]
    x = [training_encoder_input, training_decoder_input]
    y = [training_decoder_output]
    return x, y


def train_lsmt(model,
               params,
               X_train,
               X_test,
               y_train,
               y_test):
    train_input_data, train_output_data = convert_training_data(X_train.tolist(),
                                                                y_train.tolist(),
                                                                params)
    test_input_data, test_output_data = convert_training_data(X_test.tolist(),
                                                              y_test.tolist(),
                                                              params)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True,
                                                 write_images=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint('checkpoint', monitor='val_acc', verbose=1, save_best_only=True,
                                                    mode='max')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=0,
                                                      patience=2,
                                                      verbose=0, mode='auto')

    callbacks_list = [tb_callback, checkpoint, early_stopping]

    model.fit(train_input_data, train_output_data, validation_data=(test_input_data, test_output_data),
              batch_size=500, epochs=40, callbacks=callbacks_list)


def save_lstm_model(model, tf_model_name, tf_lite_model_name):
    tf.keras.models.save_model(model, tf_model_name)

    # Create a converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Convert the model
    tflite_model = converter.convert()
    # Create the tflite model file
    open(tf_lite_model_name, "wb").write(tflite_model)
