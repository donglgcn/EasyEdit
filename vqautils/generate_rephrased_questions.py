import os
import asyncio
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key="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",
)

import json

def rephrase_question(question, num_versions=10):
    # Call the OpenAI API with a prompt to generate multiple paraphrases in one go
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Please rephrase the following question in {num_versions} different ways: '{question}', without list number, seperate with new line.",
            }
        ],
        model="gpt-4-1106-preview",
    )
    # Split the response into separate paraphrases
    rephrased_questions = response.choices[0].message.content.strip().split('\n')
    # Ensure that we only get the number of versions requested
    # rephrased_questions = rephrased_questions[:num_versions]
    return rephrased_questions


def load_questions_and_rephrase(input_file, log_file="./output.log"):
    with open(input_file, 'r') as file:
        data = json.load(file)

    log = open(log_file, 'w')
    rephrased_questions_list = []
    for item in tqdm(data['questions']):
        question = item['question']
        rephrased_versions = rephrase_question(question)
        rephrased_questions_list.append({
            "image_id": item["image_id"],
            "question": question,
            "question_id": item["question_id"],
            "rephrased_questions": rephrased_versions
        })
        log.write(f"Image ID: {item['image_id']}\n")
        log.write(f"Question ID: {item['question_id']}\n")
        log.write(f"Question: {question}\n")
        log.write(f"Rephrased questions: {rephrased_versions}\n")
        log.write("\n")
    log.close()
    return rephrased_questions_list


# Example usage
input_json_file = 'OpenEnded_mscoco_val2014_questions.json'  # Replace with your JSON file path
rephrased_questions = load_questions_and_rephrase(input_json_file)

# If you need to save the rephrased questions to a new JSON file
with open('rephrased_questions.json', 'w') as outfile:
    json.dump({"rephrased_questions": rephrased_questions}, outfile, indent=4)

print(json.dumps({"rephrased_questions": rephrased_questions}, indent=4))
