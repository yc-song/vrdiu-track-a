import os
import json
import pandas as pd
import pickle
import shutil
from tqdm import tqdm
def load_pkl_files(directory):
    pkl_files = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.pkl'):
            with open(os.path.join(directory, file_name), 'rb') as file:
                pkl_files[file_name] = pickle.load(file)
    return pkl_files

def load_csv_files(directory):
    csv_files = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            csv_files[file_name] = pd.read_csv(os.path.join(directory, file_name))
    return csv_files

def create_json_for_image(image_info, label_data):
    json_data = {"form": []}
    for obj_id, obj_data in image_info['objects'].items():
        global_id = obj_data['global_id']
        try:
            label_row = label_data[label_data['label(global_id)'] == global_id]
            label = label_row['key_fix_text'].values[0] if not label_row.empty else 'NULL'
        except KeyError:
            label = 'NULL'

        form_entry = {
            "box": obj_data['bbox'],
            "text": obj_data['text'],
            "label": label,
            "words": [{
                "box": obj_data['bbox'],
                "text": obj_data['text']
            }],
            "global_id": global_id,
        }
        json_data["form"].append(form_entry)
    return json_data

def process_files(image_directory, pkl_directory, csv_directory, output_directory):


    pkl_files = load_pkl_files(pkl_directory)
    csv_files = load_csv_files(csv_directory)

    for split in ['train', 'val', 'test']:
        # pkl_file = pkl_files[f'{split}_doc_info.pkl']
        if split in ['train', 'val']:
            file_path = f'{pkl_directory}/{split}_doc_info.pkl'
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
        elif split in ['test']:
            file_path_1 = f'{pkl_directory}/{split}_doc_info.pkl'
            file_path_2 = f'{pkl_directory}/printed_doc_info.pkl'
            with open(file_path_1, 'rb') as file1, open(file_path_2, 'rb') as file2:
                data1 = pickle.load(file1)
                data2 = pickle.load(file2)
            data = {**data1, **data2}

        image_files = [f for f in os.listdir(os.path.join(image_directory, f"{split}_data/images"))]
        if split in ['train', 'val']:
            label_data = csv_files[f'{split}_dataframe.csv'] if split in ['train', 'val'] else None
        elif split in ['test']:
            label_data = csv_files['test_printed_dataframe.csv']
        if not os.path.exists(f'{output_directory}/{split}_data/annotations'):
            os.makedirs(f'{output_directory}/{split}_data/annotations')
        for image_file in tqdm(image_files):
            base_name = os.path.splitext(image_file)[0]
            json_data = create_json_for_image(data[base_name], label_data)
            output_file = os.path.join(f'{output_directory}/{split}_data/annotations', f'{base_name}.json')
            with open(output_file, 'w') as file:
                json.dump(json_data, file, indent=4)
os.makedirs(os.path.join('./data/train_data'), exist_ok= True)
os.makedirs(os.path.join('./data/val_data/'), exist_ok= True)
os.makedirs(os.path.join('./data/test_data/'), exist_ok= True)
shutil.move('./data/train_images', './data/train_data/images')
shutil.move('./data/val_images', './data/val_data/images')
shutil.move('./data/test_images' './data/test_data/images')
image_directory = './data'
pkl_directory = './data'
csv_directory = './data'
output_directory = './data'
process_files(image_directory, pkl_directory, csv_directory, output_directory)

