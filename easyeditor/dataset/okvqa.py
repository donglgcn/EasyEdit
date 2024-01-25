"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from vqautils.vqa import VQA

class OKVQADataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, debug=False, locality_root: str = None, config=None, types: typing.Union[str, typing.List[str]]=None,*args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        self.locality_root = locality_root
        super().__init__(vis_processor, vis_root, rephrase_root, [])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: "

        data = []
        # load annotations like okvqa
        # versionType ='v2_' # this should be '' when using VQA v2.0 dataset
        taskType    = 'OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
        dataSubType = 'val2014'
        annfile = os.path.join(data_dir, 'mscoco_val2014_annotations.json')
        quesfile = os.path.join(data_dir, 'OpenEnded_mscoco_val2014_questions.json')
        self.vis_root 		= os.path.join(self.vis_root, dataSubType)
        # initialize VQA api for QA annotations
        self.okvqa          = VQA(annotation_file=annfile, question_file=quesfile)
        self.annIds         = self.okvqa.getQuesIds()
        self.annotation     = self.okvqa.loadQA(self.annIds)
        if types is not None:
            if isinstance(types, str):
                types = [types]
            self.annotation = [a for a in self.annotation if a['counterfact_type'] in types]
        if size is not None:
            self.annotation = self.annotation[:size]
        for i, record in enumerate(self.annotation):
            
            if record['counterfact_answer'] == "":
                continue
            
            imgFilename = 'COCO_' + dataSubType + '_'+ str(record['image_id']).zfill(12) + '.jpg'
            image_path = os.path.join(self.vis_root, imgFilename)
            
            rephrase_image_folder = os.path.join(self.rephrase_root, str(record['question_id']))
            rephrase_image_paths = sorted([os.path.join(rephrase_image_folder, file) for file in os.listdir(rephrase_image_folder)])
            locality_image_path = os.path.join(self.locality_root, str(record['question_id']), '0.png')
            
            image = Image.open(image_path).convert("RGB")

            # debug: replace all rephrase images with the same blank image
            if debug:
                rephrase_images = [Image.open(rephrase_image_path).convert("RGB") for rephrase_image_path in rephrase_image_paths]
                # blank_img=Image.new('RGB', (364, 364), color = 'black')
                # rephrase_images = [blank_img for rephrase_image_path in rephrase_image_paths]
            else:
                rephrase_images = [Image.open(rephrase_image_path).convert("RGB") for rephrase_image_path in rephrase_image_paths]
            
            
            locality_image = Image.open(locality_image_path).convert("RGB")
            if debug:
                locality_image=locality_image # image

            image = self.vis_processor(image)
            
            processed_images = []

            # Loop through each image in the batch
            for img in rephrase_images:
                # Process each image individually using self.vis_processor
                processed_img = self.vis_processor(img)
                
                # Append the processed image to the list
                processed_images.append(processed_img)

            # Convert the list of processed images to a PyTorch tensor
            rephrase_images = torch.stack(processed_images, dim=0)

            # rephrase_images = self.vis_processor(rephrase_images)  
            locality_image = self.vis_processor(locality_image)  

            item = {
                'prompt': self.okvqa.qqa[record['question_id']]["question"],
                'pred': record["answers"][0]["answer"],
                'target': record['counterfact_answer'],
                'rephrase_prompts': self.okvqa.qqa[record['question_id']]["rephrased_questions"],
                'image': image,
                'image_rephrases': rephrase_images,
                'cond': "{} >> {} || {}".format(
                    record["answers"][0]["answer"],
                    record['counterfact_answer'],
                    self.okvqa.qqa[record['question_id']]["question"]
                )
            }
            # if debug:
            #     item['rephrase_prompts'] = ["?"]

            item['multimodal_locality_image'] = locality_image
            item['multimodal_locality_prompt'] = item['prompt']
            if debug:
                item['multimodal_locality_ground_truth'] = record['counterfact_answer']
            else:
                item['multimodal_locality_ground_truth'] = record['locality_answer']
            
            item['question_type'] = record['question_type']
            item['counterfact_type'] = record['counterfact_type']
            item['counterfact_type_reason'] = record['counterfact_type_reason']
            item['image_object'] = record['image_object']
            data.append(item)
            
        # if size is not None:
        #     data = data[:size]        
        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrases = [b['rephrase_prompts'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrases = [b['image_rephrases'] for b in batch]
        # loc_q = [b["locality_prompt"] for b in batch]
        # loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]
        
        # edit_inner
        edit_inner = {}
        edit_inner['image'] = torch.stack(image, dim=0)
        edit_inner['text_input'] = [self.prompt.format(s) + f"{t}" for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            edit_inner['labels'] = torch.cat([self.tok.encode(target, add_special_tokens=False, return_tensors="pt",) for target in trg], dim=0)
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            edit_inner['labels'] = torch.cat([self.tok.encode(target, return_tensors="pt",) for target in trg], dim=0)
        
        # edit_outer
        edit_outer = {}
        edit_outer['image'] = torch.stack(image, dim=0)
        edit_outer['text_input'] = [[self.prompt.format(r) + f"{t}" for r in rephrase] for rephrase, t in zip(rephrases, trg)]
        edit_outer['labels'] = trg
        # print("edit_outer['labels']", edit_outer['labels'])
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_outer['prompts_len'] = [[len(self.tok.encode(self.prompt.format(r), add_special_tokens=False)) for r in rephrase] for rephrase in rephrases]
            edit_outer['labels'] = torch.cat([self.tok.encode(target, add_special_tokens=False, return_tensors="pt",) for target in trg], dim=0)
            # print("edit_outer['labels'] tok", edit_outer['labels'])
        else:
            edit_outer['prompts_len'] = [[len(self.tok.encode(self.prompt.format(r))) for r in rephrase] for rephrase in rephrases]
            edit_outer['labels'] = torch.cat([self.tok.encode(target, return_tensors="pt",) for target in trg], dim=0)
            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] = torch.stack(image_rephrases, dim=0)
        edit_outer_image['text_input'] = [self.prompt.format(s) + f"{t}" for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            edit_outer_image['labels'] = torch.cat([self.tok.encode(target, add_special_tokens=False, return_tensors="pt",) for target in trg], dim=0)
        else:
            edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            edit_outer_image['labels'] = torch.cat([self.tok.encode(target, return_tensors="pt",) for target in trg], dim=0)
        
        # loc
        # "loc": "nq question: what purpose did seasonal monsoon winds have on trade", 
        # "loc_ans": "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", 
        # "m_loc": "val2014/COCO_val2014_000000297147.jpg", 
        # "m_loc_q": "What sport can you use this for?",
        # "m_loc_a": "riding"
        loc_q = ["nq question: what purpose did seasonal monsoon winds have on trade" for b in batch]
        loc_a = ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans" for b in batch]
        loc = {}
        loc['image'] = None
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
            loc['labels'] = torch.cat([self.tok.encode(loc_ans, add_special_tokens=False, return_tensors="pt",) for loc_ans in loc_a], dim=0)
        else:
            loc['prompts_len'] = [len(self.tok.encode(q)) for q in loc_q]
            loc['labels'] = torch.cat([self.tok.encode(loc_ans, return_tensors="pt",) for loc_ans in loc_a], dim=0)
        
        # m_loc
        loc_image = {}
        loc_image['image'] = torch.stack(m_loc_image, dim=0)
        loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        print("loc_image['labels']", loc_image['labels'])
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
            loc_image['labels'] = torch.cat([self.tok.encode(m_loc_ans, add_special_tokens=False, return_tensors="pt",) for m_loc_ans in m_loc_a], dim=0)
            print("loc_image['labels'] tok", loc_image['labels'])
        else:
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
            loc_image['labels'] = torch.cat([self.tok.encode(m_loc_ans, return_tensors="pt",) for m_loc_ans in m_loc_a], dim=0)
        
        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)
        
        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            "cond": cond
        }
        return dict_to(batch, self.config.device)
