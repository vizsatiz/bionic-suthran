# Introduction

The project is for creating a seq-to-seq model in Keras for adding punctuations to a string. 
Without punctuations its difficult to parse and get meaningful information from strings. This was created when I was trying to do NER
on a badly punctuated data set like text messages. 

# How to use

I have tried to make it simple to use. Just create a `config.json` file and copy the contents of `config.sample.json`.
Now, file in the options in your `config.json` as per your data and output locations.

## Config

| Config | Description |
| --- | --- |
| input_data_dir | The directory which has the data file |
| input_data_file_name | Name of the input file (Show be csv, find the format format below) |
| output_tf_model_name | Name of the output file (normal tf model) |
| output_tf_lite_model_name | Name of the output file (tf lite model) |
| network_input_size | Size of the input string (no of characters) |
| network_output_size | Size of the output string (no of characters) |
| network_enc_lstm_units | LSTM size (no of nodes) |

## Input Format

The input file should be a csv with column names `data` and `tag`. The value inside `data` should be 
string of size `network_input_size` or smaller (if bigger that will be trimmed of while training).
The value in `tag` should be properly punctuated string for the value in `data` column.

## Outputs
 The output models will be created in `build` folder with the names `output_tf_model_name` and `output_tf_lite_model_name`.
 The tf lite model can be directly integrated in android applications
 
# Whats next

- Will try to reduce the model size.
- Want to try out training with batched strings of len N < length of strings. This might help in reducing model size and increasing inference speed.

# Contributions

Any kind of valuable contributions are welcome :) 
 
 
