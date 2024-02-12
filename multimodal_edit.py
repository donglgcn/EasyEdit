import os
import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset, OKVQADataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer


def print_result(metrics):
    rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
    rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics])
    rephrase_image_acc = mean([m['post']['rephrase_image_acc'].item() for m in metrics])
    locality_acc = mean([m['post']['locality_acc'].item() for m in metrics])
    locality_image_acc = mean([m['post']['multimodal_locality_acc'].item() for m in metrics])
    print(f'rewrite_acc: {rewrite_acc}')
    print(f'rephrase_acc: {rephrase_acc}')
    print(f'rephrase_image_acc: {rephrase_image_acc}')
    print(f'locality_acc: {locality_acc}')
    print(f'multimodal_locality_acc: {locality_image_acc}')

def train_MEND_MiniGPT4_Caption():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    


def train_MEND_MiniGPT4_VQA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = VQADataset('/project/SDS/research/sds-rise/dongliang/datasets/EasyEdit/MMEDIT/editing-data-20231120T160427Z-001/editing-data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset('/project/SDS/research/sds-rise/dongliang/datasets/EasyEdit/MMEDIT/editing-data-20231120T160427Z-001/editing-data/vqa/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run() 
  
       
def train_MEND_Blip2OPT_Caption():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
        
def train_MEND_Blip2OPT_VQA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()   
     
def train_MEND_Blip2OPT_VQA_debug():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams, size=20)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams, size=20)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    
    
def train_MEND_Blip2OPT_VQA_Vision_debug():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2-vision.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams, size=20)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams, size=20)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()  
      
def train_MEND_Blip2OPT_VQA_Vision():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2-vision.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    
    
def test_MEND_MiniGPT4_VQA():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('/project/SDS/research/sds-rise/dongliang/datasets/EasyEdit/MMEDIT/editing-data-20231120T160427Z-001/editing-data/vqa/vqa_eval.json',
                         debug=True, config=hparams)
    hparams.results_dir='./results/test_MEND_MiniGPT4_VQA_DEBUG_blackimage'
    os.makedirs(hparams.results_dir, exist_ok=True)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()    


def test_MEND_MiniGPT4_OKVQA():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    hparams.rephrase_image = '/project/SDS/research/sds-rise/dongliang/datasets/VQA/rephrased_images/'
    eval_ds = OKVQADataset('vqautils', locality_root='/project/SDS/research/sds-rise/dongliang/datasets/VQA/locality_images_dalle2/', 
                           config=hparams)    
    
    hparams.results_dir='./results/test_MEND_MiniGPT4_OKVQA_locality_label'
    os.makedirs(hparams.results_dir, exist_ok=True)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()


def train_SERAC_MiniGPT4_Caption():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/minigpt4.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
    
def train_SERAC_MiniGPT4_Caption_debug():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/minigpt4.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams, size=5)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams, size=5)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
    
def train_SERAC_MiniGPT4_VQA():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/minigpt4.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
  
    
def train_SERAC_Blip2OPT_Caption():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/blip2.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
    
def train_SERAC_Blip2OPT_VQA():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/blip2.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
    
def train_SERAC_Blip2OPT_Caption_debug():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/blip2.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams, size=20)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams, size=20)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()


def test_SERAC_MiniGPT4_Caption():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    # train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit_test.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()
  
    
def edit_SERAC_MiniGPT4_Caption():
    prompts = [
        "a photo of",
        "a photo of"
    ]
    targets = [
        "A couple trays of cookies on a counter.",
        "a couple of people that are cutting a piece of cake",
    ]
    image = [
        "val2014/COCO_val2014_000000575018.jpg",
        "val2014/COCO_val2014_000000048332.jpg"
    ]
    rephrase_prompts = [
        "a photograph of",
        "give a detailed description of the picture,"
    ]
    rephrase_image = [
        "val2014_image_rephrase/COCO_val2014_000000575018.png",
        "val2014_image_rephrase/COCO_val2014_000000048332.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        keep_original_weight=True        
    )


def edit_SERAC_Blip2OPT_Caption():
    prompts = [
        "a photo of",
        "a photo of"
    ]
    targets = [
        "A couple trays of cookies on a counter.",
        "a couple of people that are cutting a piece of cake",
    ]
    image = [
        "val2014/COCO_val2014_000000575018.jpg",
        "val2014/COCO_val2014_000000048332.jpg"
    ]
    rephrase_prompts = [
        "a photograph of",
        "give a detailed description of the picture,"
    ]
    rephrase_image = [
        "val2014_image_rephrase/COCO_val2014_000000575018.png",
        "val2014_image_rephrase/COCO_val2014_000000048332.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        keep_original_weight=True        
    )


def edit_IKE_MiniGPT4_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('/localtmp/ktm8eh/datasets/EasyEdit/MMEDIT/editing-data-20231120T160427Z-001/editing-data/vqa/vqa_train.json', config=hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        train_ds=train_ds,
        keep_original_weight=True        
    )


def edit_IKE_MiniGPT4_Caption():
    prompts = [
        "a photo of",
        "a photo of"
    ]
    targets = [
        "A selection of wooden kitchen tools on a counter.",
        "Bicyclists on a city street, most not using the bike lane",
    ]
    image = [
        "val2014/COCO_val2014_000000386164.jpg",
        "val2014/COCO_val2014_000000462565.jpg"
    ]
    rephrase_prompts = [
        "provide a brief overview of the image content,",
        "describe the image content,"
    ]
    rephrase_image = [
        "val2014_image_rephrase/COCO_val2014_000000386164.png",
        "val2014_image_rephrase/COCO_val2014_000000462565.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('/localtmp/ktm8eh/datasets/EasyEdit/MMEDIT/editing-data-20231120T160427Z-001/editing-data/vqa/vqa_train.json', config=hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        train_ds=train_ds,
        keep_original_weight=True        
    )


def edit_IKE_Blip2OPT_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
def test_IKE_Blip2OPT_Caption():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)

def test_IKE_Blip2OPT_VQA():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)
    
def Generate_Embedding_for_IKE():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    train_ds = VQADataset('/localtmp/ktm8eh/datasets/EasyEdit/MMEDIT/editing-data-20231120T160427Z-001/editing-data/vqa/vqa_train.json', config=hparams)
    ## Generate embedding files for IKE
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    encode_ike_facts_multimodal(sentence_model, train_ds, hparams)
    
def test_IKE_MiniGPT4_Caption():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('/localtmp/ktm8eh/datasets/EasyEdit/MMEDIT/editing-data-20231120T160427Z-001/editing-data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset('/localtmp/ktm8eh/datasets/EasyEdit/MMEDIT/editing-data-20231120T160427Z-001/editing-data/vqa/vqa_eval.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)
    
def test_IKE_MiniGPT4_VQA_debug():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams, size=5)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams, size=5)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)
    
def test_IKE_MiniGPT4_VQA():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    # train_ds = VQADataset('data/vqa_train.json', config=hparams, size=5)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)
    
def test_IKE_Blip2OPT_VQA_debug():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams, size=100)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams, size=100)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)
    
def edit_MEND_MiniGPT4_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        keep_original_weight=True        
    )
    print(metrics)
    
    
def edit_SERAC_MiniGPT4_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        train_ds=train_ds,
        keep_original_weight=True        
    )
 
 
def edit_MEND_MiniGPT4_Caption():
    prompts = [
        "a photo of",
        "a photo of"
    ]
    targets = [
        "A selection of wooden kitchen tools on a counter.",
        "Bicyclists on a city street, most not using the bike lane",
    ]
    image = [
        "val2014/COCO_val2014_000000386164.jpg",
        "val2014/COCO_val2014_000000462565.jpg"
    ]
    rephrase_prompts = [
        "provide a brief overview of the image content,",
        "describe the image content,"
    ]
    rephrase_image = [
        "val2014_image_rephrase/COCO_val2014_000000386164.png",
        "val2014_image_rephrase/COCO_val2014_000000462565.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        keep_original_weight=True        
    )


def test_IKE_MiniGPT4_OKVQA():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('/project/SDS/research/sds-rise/dongliang/datasets/EasyEdit/MMEDIT/editing-data-20231120T160427Z-001/editing-data/vqa/vqa_train.json', config=hparams)
    hparams.rephrase_image = '/project/SDS/research/sds-rise/dongliang/datasets/VQA/rephrased_images/'
    eval_ds = OKVQADataset('vqautils', debug=True, locality_root='/project/SDS/research/sds-rise/dongliang/datasets/VQA/locality_images_dalle2/', 
                           config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    #dump metrics
    import pickle
    with open('okvqa_metrics_debug_blackimage.pkl', 'wb') as f:
        pickle.dump(metrics, f)

def load_metrics(path):
    import pickle
    with open(path, 'rb') as f:
        metrics = pickle.load(f)
    post_rewrite_acc = mean([m['post']['rewrite_acc'] for m in metrics])
    post_rephrase_acc = mean([m['post']['rephrase_acc'] for m in metrics])
    post_rephrase_image_acc = mean([m['post']['rephrase_image_acc'] for m in metrics])
    pre_rewrite_acc = mean([m['pre']['rewrite_acc'] for m in metrics])
    pre_rephrase_acc = mean([m['pre']['rephrase_acc'] for m in metrics])
    pre_rephrase_image_acc = mean([m['pre']['rephrase_image_acc'] for m in metrics])
    # print results
    print(f'post_rewrite_acc: {post_rewrite_acc}\npost_rephrase_acc: {post_rephrase_acc}\npost_rephrase_image_acc: {post_rephrase_image_acc}\npre_rewrite_acc: {pre_rewrite_acc}\npre_rephrase_acc: {pre_rephrase_acc}\npre_rephrase_image_acc: {pre_rephrase_image_acc}')
    return
    
if __name__ == "__main__":
    
    # train_MEND_MiniGPT4_Caption()
    # train_MEND_MiniGPT4_VQA()
    # train_MEND_Blip2OPT_Caption()
    # train_MEND_Blip2OPT_VQA()
    # train_MEND_Blip2OPT_VQA_Vision()
    # train_MEND_Blip2OPT_VQA_debug()
    # train_MEND_Blip2OPT_VQA_Vision_debug()
    
    
    # train_SERAC_MiniGPT4_Caption()
    # train_SERAC_MiniGPT4_Caption_debug()
    # train_SERAC_MiniGPT4_VQA()
    # train_SERAC_Blip2OPT_Caption()
    # train_SERAC_Blip2OPT_VQA()
    # train_SERAC_Blip2OPT_Caption_debug()
    
    
    # test_SERAC_MiniGPT4_Caption()
    # test_MEND_MiniGPT4_VQA()
    test_MEND_MiniGPT4_OKVQA()
    # Generate_Embedding_for_IKE()
    # test_IKE_MiniGPT4_Caption()
    # test_IKE_MiniGPT4_VQA()
    # test_IKE_MiniGPT4_VQA_debug()
    # test_IKE_Blip2OPT_VQA()
    # test_IKE_MiniGPT4_VQA()
    # test_IKE_Blip2OPT_VQA_debug()
    # test_IKE_MiniGPT4_OKVQA()
    

    # edit_MEND_MiniGPT4_Caption()
    # edit_MEND_MiniGPT4_VQA()
    # edit_SERAC_MiniGPT4_Caption()
    # edit_SERAC_Blip2OPT_Caption()
    # edit_IKE_MiniGPT4_Caption()
    # edit_IKE_MiniGPT4_VQA()
    # edit_IKE_Blip2OPT_VQA()

    # load_metrics("okvqa_metrics_debug_blackimage.pkl")
