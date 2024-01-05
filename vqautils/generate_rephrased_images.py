import torch
import json
import os

from tqdm import tqdm
os.environ['HF_HOME'] = '/localtmp/ktm8eh/.cache/huggingface'
IMAGE_ROOT = '/localtmp/ktm8eh/datasets/VQA/rephrased_images/'
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def load_diffusion_model(model_id="stabilityai/stable-diffusion-2-1", device="cuda" if torch.cuda.is_available() else "cpu"):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32,)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.requires_safety_checker = False
    pipe.safety_checker = None
    pipe = pipe.to(device)
    return pipe

def load_prompt(prompt_file='./mscoco_val2014_annotations.json', partition=0, every=505):
    with open(prompt_file, 'r') as file:
        prompt = json.load(file)
    anns = prompt['annotations'][partition*every: (partition+1)*every]
    return anns

def generate_images(anns, pipe, num_images_per_prompt=10, image_root=IMAGE_ROOT):
    with torch.no_grad():
        for ann in anns:
            question_id = ann['question_id']
            prompt = ann['image_object']
            dir = os.path.join(image_root, str(question_id))
            if os.path.exists(dir):
                continue
            os.makedirs(dir, exist_ok=True)
            index=0
            for _ in range(2):
                images = pipe(prompt, num_images_per_prompt=num_images_per_prompt//2).images
                for image in images:
                    image.save(os.path.join(dir, f'{index}.jpg'), format='JPEG')
                    index += 1
            print(f"Saved {num_images_per_prompt} images for question {question_id}")
    return

def main(args):
    pipe = load_diffusion_model()
    anns = load_prompt(partition=args.partition, every=args.every)
    generate_images(anns, pipe)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=int, default=0)
    parser.add_argument('--every', type=int, default=253)
    args = parser.parse_args()
    main(args)
