import transformers
import sys
sys.path.append("../easyeditor/dataset/processor/")
# from blip_processors import BlipImageEvalProcessor
# from easyeditor import MiniGPT4
from vqa import VQA
import random
import matplotlib.pyplot as plt
import os
import torch
from PIL import Image

ROOT		='/media/dongliang/10TB Disk/datasets/VQA/okvqa'
# versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataSubType ='val2014'
annfile = os.path.join(ROOT, 'mscoco_val2014_annotations.json')
quesfile = os.path.join(ROOT, 'OpenEnded_mscoco_val2014_questions.json')
imgDir 		= os.path.join("/media/dongliang/10TB Disk/datasets/mscoco/images", dataSubType)

# initialize VQA api for QA annotations
vqa=VQA(annotation_file=annfile, question_file=quesfile)

# load and display QA annotations for given question types
"""
All possible quesTypes for abstract and mscoco has been provided in respective text files in ../QuestionTypes/ folder.
"""
# annIds = vqa.getQuesIds(quesTypes='six')
annIds = vqa.getQuesIds()
anns = vqa.loadQA(annIds)[3768:3770]
while True:
    randomAnn = random.choice(anns)
    vqa.showQA([randomAnn])
    imgId = randomAnn['image_id']
    imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
    if os.path.isfile(os.path.join(imgDir, imgFilename)):
        I = plt.imread(os.path.join(imgDir, imgFilename))
        plt.imshow(I)
        plt.axis('off')
        plt.show()

# import torch
# from PIL import Image
# from lavis.models import load_model_and_preprocess
# pip install transformers==4.26
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="okvqa", is_eval=True, device=device)
# # ask a random question.
# question = "Which city is this photo taken?"
# raw_image = Image.open(os.path.join(imgDir, imgFilename)).convert("RGB")
# image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# question = txt_processors["eval"](question)
# ans = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
# print(ans)
#
# from easyeditor import MiniGPT4
# qformer_checkpoint= "../hugging_cache/blip2_pretrained_flant5xxl.pth"
# qformer_name_or_path= "bert-base-uncased"
# state_dict_file= "../hugging_cache/eva_vit_g.pth"
# name= "../hugging_cache/vicuna-13b-v1.5"
# model_name= "minigpt4"
# model = MiniGPT4(
#     vit_model="eva_clip_g",
#     q_former_model=qformer_checkpoint,
#     img_size=364,
#     use_grad_checkpoint=False,
#     vit_precision="int8",
#     freeze_vit=True,
#     llama_model=name,
#     state_dict_file=state_dict_file,
#     qformer_name_or_path=qformer_name_or_path,
#     device_8bit="cpu",
# )
# vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
# question = ["Which city is this photo taken?"]
# raw_image = Image.open(os.path.join(imgDir, imgFilename)).convert("RGB")
# image = vis_processor(raw_image).unsqueeze(0).to('cpu')
# ans = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
# print(ans)
# input()
#
