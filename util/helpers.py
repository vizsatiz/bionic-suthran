import os
import json
import io


def get_all_file_names_in_dir(folder_path, file_type):
    return [x for x in os.listdir(folder_path) if x.endswith(file_type)]


def get_json_from_json_file(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def write_json_data_to_file(file_path, data, indent=2, minify=False):
    with open(file_path, "w+") as f:
        if not minify:
            json.dump(data, f, indent=indent)
        else:
            json.dump(data, f, separators=(',', ":"))
