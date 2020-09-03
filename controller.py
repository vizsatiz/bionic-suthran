from sklearn.model_selection import train_test_split

from config import input_dir, input_file, input_size, lstm_size
from model import build_params, build_model
from predict import infer
from train import train_lsmt, save_lstm_model
import tensorflow as tf


def setup_test_and_train_set(in_file: str):
    import pandas as pd
    com_data = pd.read_csv(in_file)
    input_data = com_data['data']
    output_data = com_data['tag']
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.3, random_state=42)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    return x_train, x_test, y_train, y_test


def setup_model(x_train, y_train):
    build_params(input_data=x_train,
                 output_data=y_train,
                 params_path="./build/param",
                 max_lenghts=input_size)
    model, params = build_model(params_path="./build/param",
                                enc_lstm_units=lstm_size)
    return model, params


def train_and_save_model():
    x_train, x_test, y_train, y_test = setup_test_and_train_set("{}/{}".format(input_dir, input_file))
    model, params = setup_model(x_train, y_train)
    train_lsmt(model, params, x_train, x_test, y_train, y_test)
    save_lstm_model(model, "final_model.wb", "final_tflite.rt")


def infer_from_model(model_name, messages):
    _, params = build_model(params_path="./build/param",
                            enc_lstm_units=128)
    model = tf.keras.models.load_model("./build/{}".format(model_name))
    return infer(messages, model, params)


if __name__ == '__main__':
    train_and_save_model()
