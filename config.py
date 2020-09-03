import json
import os

with open("config.json", 'r') as config:
    config_json = json.load(config)

if not os.path.exists('build'):
    os.makedirs('build')

input_dir = config_json['input_data_dir']
input_file = config_json['input_data_file_name']
input_size = (config_json['network_input_size'], config_json['network_output_size'])

lstm_size = config_json['network_enc_lstm_units']

output_tf_model_name = config_json['output_tf_model_name']
output_tf_lite_model_name = config_json['output_tf_lite_model_name']
