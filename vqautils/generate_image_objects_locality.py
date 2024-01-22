import os
import asyncio
from tqdm import tqdm
from openai import OpenAI
from KEY import API_KEY

client = OpenAI(
    # This is the default and can be omitted
    api_key=API_KEY,
)
import json

def predict_image_object(question, answer):
    # Call the OpenAI API with a prompt to generate multiple paraphrases in one go
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Given ('question': '{question}', 'answer': '{answer}'), what object should be in the image? Short answer.\nThe objects in the image should be ",
            }
        ],
        model="gpt-4-1106-preview",
    )
    # Split the response into separate paraphrases
    image_object = response.choices[0].message.content.strip().rstrip('.')
    # Ensure that we only get the number of versions requested
    # rephrased_questions = rephrased_questions[:num_versions]
    return image_object


def load_questions_and_rephrase(vqa):
    annIds = vqa.getQuesIds()
    anns = vqa.loadQA(annIds)
    index = 0
    save_every = 1000
    rephrased_anns_list = []
    flag = False
    for i, ann in tqdm(enumerate(anns)):
        if ann.get('locality_image_object', None) is not None:
            continue
        flag=False
        quesId = ann["question_id"]
        question = vqa.qqa[quesId]["question"]
        answer = ann["locality_answer"]
        image_object = predict_image_object(question, answer)
        ann["locality_image_object"] = image_object
        rephrased_anns_list.append(ann)

        print(f"pred_locality_image_object: {image_object}\n")
        if (i+1) % save_every == 0:
            with open(f'output_{index}.json', 'w') as outfile:
                json.dump({"pred_locality_image_object": rephrased_anns_list[i-save_every+1:i+1]}, outfile, indent=4)
            index += 1
            flag = True
    if not flag:
        with open(f'output_{index}.json', 'w') as outfile:
            json.dump({"pred_locality_image_object": rephrased_anns_list[index * save_every:]}, outfile, indent=4)
    return rephrased_anns_list



# Example usage
ROOT		='./vqautils'
# versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataSubType ='val2014'
annfile = os.path.join(ROOT, 'mscoco_val2014_annotations.json')
quesfile = os.path.join(ROOT, 'OpenEnded_mscoco_val2014_questions.json')
imgDir 		= os.path.join("/media/dongliang/10TB Disk/datasets/mscoco/images", dataSubType)

# train OKVQA dataset
# ROOT		='/localtmp/ktm8eh/datasets/VQA/okvqa'
# # versionType ='v2_' # this should be '' when using VQA v2.0 dataset
# taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
# dataSubType ='train2014'
# annfile = os.path.join(ROOT, 'mscoco_train2014_annotations.json')
# quesfile = os.path.join(ROOT, 'OpenEnded_mscoco_train2014_questions.json')
# imgDir 		= os.path.join("/localtmp/ktm8eh/datasets/mscoco/images", dataSubType)
from vqa import VQA
# initialize VQA api for QA annotations
vqa=VQA(annotation_file=annfile, question_file=quesfile)

rephrased_questions = load_questions_and_rephrase(vqa)

# If you need to save the rephrased questions to a new JSON file
with open('pred_locality_image_object.json', 'w') as outfile:
    json.dump({"pred_locality_image_object": rephrased_questions}, outfile, indent=4)

# print(json.dumps({"pred_image_object": rephrased_questions}, indent=4))
