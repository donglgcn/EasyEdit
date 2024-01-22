# coding: utf-8

from vqa import VQA
import random
import matplotlib.pyplot as plt
import os

ROOT		='/localtmp/ktm8eh/datasets/VQA/okvqa'
# versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataSubType ='val2014'
annfile = os.path.join(ROOT, 'mscoco_val2014_annotations.json')
quesfile = os.path.join(ROOT, 'OpenEnded_mscoco_val2014_questions.json')
imgDir 		= os.path.join("/media/dongliang/10TB Disk/datasets/mscoco/images", dataSubType)

# train OKVQA dataset
ROOT		='/localtmp/ktm8eh/datasets/VQA/okvqa'
# versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataSubType ='train2014'
annfile = os.path.join(ROOT, 'mscoco_train2014_annotations.json')
quesfile = os.path.join(ROOT, 'OpenEnded_mscoco_train2014_questions.json')
imgDir 		= os.path.join("/localtmp/ktm8eh/datasets/mscoco/images", dataSubType)

# initialize VQA api for QA annotations
vqa=VQA(annotation_file=annfile, question_file=quesfile)

# load and display QA annotations for given question types
"""
All possible quesTypes for abstract and mscoco has been provided in respective text files in ../QuestionTypes/ folder.
"""
annIds = vqa.getQuesIds(quesTypes='six')
anns = vqa.loadQA(annIds)
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])
imgId = randomAnn['image_id']
imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
if os.path.isfile(os.path.join(imgDir, imgFilename)):
	I = plt.imread(os.path.join(imgDir, imgFilename))
	plt.imshow(I)
	plt.axis('off')
	plt.show()

# load and display QA annotations for given answer types
"""
ansTypes can be one of the following
yes/no
number
other
"""
annIds = vqa.getQuesIds(ansTypes='other')
anns = vqa.loadQA(annIds)
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])
imgId = randomAnn['image_id']
imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
if os.path.isfile(os.path.join(imgDir, imgFilename)):
	I = plt.imread(os.path.join(imgDir, imgFilename))
	plt.imshow(I)
	plt.axis('off')
	plt.show()

# load and display QA annotations for given images
"""
Usage: vqa.getImgIds(quesIds=[], quesTypes=[], ansTypes=[])
Above method can be used to retrieve imageIds for given question Ids or given question types or given answer types.
"""
ids = vqa.getImgIds()
annIds = vqa.getQuesIds(imgIds=random.sample(ids,5));  
anns = vqa.loadQA(annIds)
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])  
imgId = randomAnn['image_id']
imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
if os.path.isfile(os.path.join(imgDir, imgFilename)):
	I = plt.imread(os.path.join(imgDir, imgFilename))
	plt.imshow(I)
	plt.axis('off')
	plt.show()

