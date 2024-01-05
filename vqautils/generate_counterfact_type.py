import os
import asyncio
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key="",
)
"""
Given ('question': 'What popular company makes the computer in the forefront of the image?', 'image object': 'a dell computer', 'answer': 'dell', 'counterfact answer': 'apple'), which type is the conterfact changes, Text based or image-text based?
[image-text based], because the changed fact is that the name of the image object. <image, is, dell> to <image, is , apple>
Given ('question': 'What country serves this msk-ost?', 'image object': 'boeing plane', 'answer': 'United States', 'counterfact answer': 'China'), which type is the conterfact changes, Text based or image-text based?
[text base], because the changed fact is text common sense. <image, is, boeing>, <boeing, from, United States> to <image, is, boeing>, <boeing, from, China> and the name of the image object is not changed
Given ('question': 'What day might I most commonly go to this building', 'image object': 'a church', 'answer': 'Sunday', 'counterfact answer': 'Saturday'), which type is the conterfact changes, Text based or image-text based?
"""

import json

def predict_counterfact_type(question, answer, image_object, counterfact_answer):
    # Call the OpenAI API with a prompt to generate multiple paraphrases in one go
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Given ('question': 'What popular company makes the computer in the forefront of the image?', 'image object': 'a dell computer', 'answer': 'dell', 'counterfact answer': 'apple'), which type is the conterfact changes, Text based or image-text based?",
            },
            {
                "role": "assistant",
                "content": "[image-text based], because the changed fact is the identification of the image object. <image, is, dell> to <image, is, apple>",
            },
            {
                "role": "user",
                "content": "Given ('question': 'What country serves this most?', 'image object': 'boeing plane', 'answer': 'United States', 'counterfact answer': 'China'), which type is the conterfact changes, Text based or image-text based?",
            },
            {
                "role": "assistant",
                "content": "[text base], because the changed fact is text common sense. <image, is, boeing>, <boeing, from, United States> to <image, is, boeing>, <boeing, from, China> and the name of the image object is not changed",
            },
            {
                "role": "user",
                "content": f"Given ('question': '{question}', 'image object': '{image_object}','answer': '{answer}', 'counterfact answer': {counterfact_answer}), which type is the conterfact changes, Text based or image-text based?",
            }
        ],
        model="gpt-4-1106-preview",
    )
    # Split the response into separate paraphrases
    counterfact_answer = response.choices[0].message.content.strip()
    return counterfact_answer


def load_questions_and_rephrase(vqa):
    annIds = vqa.getQuesIds()
    anns = vqa.loadQA(annIds)
    index = 0
    save_every = 500
    rephrased_anns_list = []
    flag = False
    for i, ann in tqdm(enumerate(anns)):
        if ann.get('counterfact_type', None) is not None:
            continue
        flag=False
        quesId = ann["question_id"]
        question = vqa.qqa[quesId]["question"]
        answer = ann["answers"][0]["answer"]
        image_object = ann["image_object"]
        counterfact_answer = ann["counterfact_answer"]
        counterfact_type = predict_counterfact_type(question, answer, image_object, counterfact_answer)
        ann["counterfact_type_reason"] = counterfact_type
        # re to select content in ['']
        counterfact_type = counterfact_type[counterfact_type.find('[')+1:counterfact_type.find(']')]
        ann["counterfact_type"] = counterfact_type
        rephrased_anns_list.append(ann)

        print(f"counterfact_type: {counterfact_type}\n")
        if (i+1) % save_every == 0:
            with open(f'output_{index}.json', 'w') as outfile:
                json.dump({"counterfact_type": rephrased_anns_list[i-save_every+1:i+1]}, outfile, indent=4)
            index += 1
            flag = True
    if not flag:
        with open(f'output_{index}.json', 'w') as outfile:
            json.dump({"counterfact_type": rephrased_anns_list[index * save_every:]}, outfile, indent=4)
    return rephrased_anns_list



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

rephrased_anns_list = load_questions_and_rephrase(vqa)

# If you need to save the rephrased questions to a new JSON file
with open('counterfact_type_new.json', 'w') as outfile:
    json.dump({"counterfact_type": rephrased_anns_list}, outfile, indent=4)

# print(json.dumps({"pred_image_object": rephrased_questions}, indent=4))
