import json
import os
import re

from tqdm import tqdm

def clean_rephrased_questions(data):
    for item in data['rephrased_questions']:
        # Filter out empty strings from the rephrased questions
        item['rephrased_questions'] = [q for q in item['rephrased_questions'] if q not in ['', ' ', None]]
        if len(item['rephrased_questions']) != 10:
            print(f"Question ID: {item['question_id']}")
            print(f"Question: {item['question']}")
            print("\n")
            input()

    return data

def load_rephrased_questions(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    return data

def save_rephrased_questions(data, output_file):
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def purge_rephrased_questions(input_file, output_file):
    data = load_rephrased_questions(input_file)
    data = clean_rephrased_questions(data)
    original_data = load_rephrased_questions('OpenEnded_mscoco_val2014_questions.json')
    original_data['questions'] = data['rephrased_questions']
    save_rephrased_questions(original_data, output_file)

"""
loading VQA annotations and questions into memory...
creating index...
index created!
1it [00:01,  1.39s/it]pred_image_object: running shoes or a racing bicycle

2it [00:02,  1.38s/it]pred_image_object: a vine or climbing plant

3it [00:03,  1.03s/it]pred_image_object: a stuffed animal

pred_image_object: cards

5it [00:04,  1.19it/s]pred_image_object: a red bag containing cloth

6it [00:05,  1.04it/s]pred_image_object: a dirty or unflushed toilet

pred_image_object: a kitchen island

8it [00:08,  1.15s/it]pred_image_object: shopping bags, carts, baskets, or products/items for sale

9it [00:09,  1.11s/it]pred_image_object: plants or trees

pred_image_object: a bat and a man"""
def purge_output_file(log_file):
    with open(log_file, 'r') as file:
        log_data = file.read()
    matches = re.findall(r'pred_image_object: (.+?)(?=\n|$)', log_data)

    # The 'matches' list now contains all the extracted objects
    return matches

# merge two json file
def merge_image_object_json(image_object, original_data):
    with open(image_object, 'r') as file:
        data1 = json.load(file)
    with open(original_data, 'r') as file:
        data2 = json.load(file)
    index = 0
    data2['annotations'] = data1['pred_locality_image_object']
    
    #f dump
    with open('./vqautils/output.json', 'w') as outfile:
        json.dump(data2, outfile, indent=4)
    return data1

def rename_folder(original_data = './vqautils/mscoco_val2014_annotations.json', root = '/localtmp/ktm8eh/datasets/VQA/rephrased_images/'):
    with open(original_data, 'r') as file:
        anns = json.load(file)['annotations']
        for ann in anns:
            # Strip any whitespace and split the line into old and new names
            old_name, new_name = str(ann['image_id']), str(ann['question_id'])
            old_name = os.path.join(root, old_name)
            new_name = os.path.join(root, new_name)
            # Check if the old folder name exists
            if not os.path.exists(new_name):
                if os.path.isdir(old_name):
                    # Rename the folder
                    os.rename(old_name, new_name)
                    print(f"Renamed {old_name} to {new_name}")
                else:
                    print(f"Directory {old_name} does not exist")

def compare_jsons(json1, json2, json3):
    # Example usage
    ROOT		='./vqautils'
    # versionType ='v2_' # this should be '' when using VQA v2.0 dataset
    taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
    dataSubType ='val2014'
    annfile = os.path.join(ROOT, 'mscoco_val2014_annotations.json')
    quesfile = os.path.join(ROOT, 'OpenEnded_mscoco_val2014_questions.json')
    imgDir 		= os.path.join("/media/dongliang/10TB Disk/datasets/mscoco/images", dataSubType)
    from vqa import VQA
    # initialize VQA api for QA annotations
    vqa=VQA(annotation_file=annfile, question_file=quesfile)
    with open(json1, 'r') as file:
        data1 = json.load(file)
    with open(json2, 'r') as file:
        data2 = json.load(file)
    with open(json3, 'r') as file:
        data3 = json.load(file)
    data = []
    annIds = vqa.getQuesIds()
    anns = vqa.loadQA(annIds)
    for i, ann in tqdm(enumerate(anns)):
        flag=False
        quesId = ann["question_id"]
        question = vqa.qqa[quesId]["question"]
        answer = ann["answers"][0]["answer"]
        image_object = ann["image_object"]
        counterfact_answer = ann["counterfact_answer"]
        counterfact_type_str = data1['counterfact_type'][i]['counterfact_type'].lower()
        if 'image-text' in counterfact_type_str or 'image text' in counterfact_type_str:
            # Do something
            data1['counterfact_type'][i]['counterfact_type'] = 'image-text based'
        else:
            data1['counterfact_type'][i]['counterfact_type'] = 'text based'
        counterfact_type_str = data2['counterfact_type'][i]['counterfact_type'].lower()
        if 'image-text' in counterfact_type_str or 'image text' in counterfact_type_str:
            # Do something
            data2['counterfact_type'][i]['counterfact_type'] = 'image-text based'
        else:
            data2['counterfact_type'][i]['counterfact_type'] = 'text based'
        counterfact_type_str = data3['counterfact_type'][i]['counterfact_type'].lower()
        if 'image-text' in counterfact_type_str or 'image text' in counterfact_type_str:
            # Do something
            data3['counterfact_type'][i]['counterfact_type'] = 'image-text based'
        else:
            data3['counterfact_type'][i]['counterfact_type'] = 'text based'
        # compare 3 jsons to determine the correct counterfact_type
        types = [data1['counterfact_type'][i]['counterfact_type'], data2['counterfact_type'][i]['counterfact_type'], data3['counterfact_type'][i]['counterfact_type']]
        if types.count('image-text based') > types.count('text based'):
            if data1['counterfact_type'][i]['counterfact_type'] == 'image-text based':
                data.append(data1['counterfact_type'][i])
            elif data2['counterfact_type'][i]['counterfact_type'] == 'image-text based':
                data.append(data2['counterfact_type'][i])
            else:
                data.append(data3['counterfact_type'][i])
        else:
            if data1['counterfact_type'][i]['counterfact_type'] == 'text based':
                data.append(data1['counterfact_type'][i])
            elif data2['counterfact_type'][i]['counterfact_type'] == 'text based':
                data.append(data2['counterfact_type'][i])
            else:
                data.append(data3['counterfact_type'][i])
    with open('counterfact_type_merge.json', 'w') as outfile:
        json.dump({"counterfact_type": data}, outfile, indent=4)


def purge_locality_answer(locality_file):
    with open(locality_file, 'r') as file:
        data = json.load(file)
    for i, item in enumerate(data['pred_locality_answer']):
        # Filter out empty strings from the rephrased questions
        item['locality_answer'] = item['locality_answer'].strip("'")
        item['locality_answer'] = item['locality_answer'].strip('"')
        data['pred_locality_answer'][i] = item
    with open('locality_file_purge.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)
    return data

if __name__ == '__main__':
    # compare_jsons('vqautils/counterfact_type_new.json', 'vqautils/counterfact_type.json', 'vqautils/counterfact_type_previous.json')
    # rename_folder()
    # purge_rephrased_questions('rephrased_questions.json', 'rephrased_questions_purged.json')
    # purge_locality_answer('./locality_answer.json')
    merge_image_object_json('./pred_locality_image_object.json', './vqautils/mscoco_val2014_annotations.json')