import os
import asyncio
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key="",
)
# response = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": "Given ('question': 'What popular company makes the computer in the forefront of the image?', 'answer': 'dell'), what object should be in the image? Short answer.\nThe objects in the image should be ",
#             }
#         ],
#         model="gpt-4-1106-preview",
#     )
#     # Split the response into separate paraphrases
# rephrased_questions = response.choices[0].message.content.strip().rstrip('.')
import json

def predict_counterfact_answer(question, answer):
    # Call the OpenAI API with a prompt to generate multiple paraphrases in one go
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Given ('question': 'What popular company makes the computer in the forefront of the image?', 'answer': 'dell'), what could be one of the counterfact answer? Short answer.\nThe counterfact answer could be []?",
            },
            {
                "role": "assistant",
                "content": "[apple]",
            },
            {
                "role": "user",
                "content": f"Given ('question': '{question}', 'answer': '{answer}'), what could be one of the counterfact answer? Short answer.\nThe counterfact answer could be []?",
            }
        ],
        model="gpt-4-1106-preview",
    )
    # Split the response into separate paraphrases
    counterfact_answer = response.choices[0].message.content.strip()
    # re to select content in ['']
    counterfact_answer = counterfact_answer[counterfact_answer.find('[')+1:counterfact_answer.find(']')]
    return counterfact_answer


def load_questions_and_rephrase(vqa):
    annIds = vqa.getQuesIds()
    anns = vqa.loadQA(annIds)
    index = 0
    save_every = 500
    rephrased_anns_list = []
    flag = False
    for i, ann in tqdm(enumerate(anns)):
        if ann.get('counterfact_answer', None) is not None:
            continue
        flag=False
        quesId = ann["question_id"]
        question = vqa.qqa[quesId]["question"]
        answer = ann["answers"][0]["answer"]
        counterfact_answer = predict_counterfact_answer(question, answer)
        ann["counterfact_answer"] = counterfact_answer
        rephrased_anns_list.append(ann)

        print(f"pred_counterfact_answer: {counterfact_answer}\n")
        if (i+1) % save_every == 0:
            with open(f'output_{index}.json', 'w') as outfile:
                json.dump({"pred_counterfact_answer": rephrased_anns_list[i-save_every+1:i+1]}, outfile, indent=4)
            index += 1
            flag = True
    if not flag:
        with open(f'output_{index}.json', 'w') as outfile:
            json.dump({"pred_counterfact_answer": rephrased_anns_list[index * save_every:]}, outfile, indent=4)
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
with open('counterfact_answer.json', 'w') as outfile:
    json.dump({"pred_counterfact_answer": rephrased_anns_list}, outfile, indent=4)

# print(json.dumps({"pred_image_object": rephrased_questions}, indent=4))
