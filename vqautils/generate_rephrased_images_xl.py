import torch
import json
import os

from tqdm import tqdm
os.environ['HF_HOME'] = '/localtmp/ktm8eh/.cache/huggingface'
IMAGE_ROOT = '/localtmp/ktm8eh/datasets/VQA/rephrased_images/'
from diffusers import AutoPipelineForText2Image

def load_diffusion_xl_model():
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    return pipe



def load_prompt(prompt_file='./mscoco_val2014_annotations.json', partition=0, every=505):
    with open(prompt_file, 'r') as file:
        prompt = json.load(file)
    anns = prompt['annotations'][partition*every: (partition+1)*every]
    return anns

def generate_images(anns, pipe, num_images_per_prompt=10, image_root=IMAGE_ROOT):
    with torch.no_grad():
        n_steps = 2
        # high_noise_frac = 0.8
        for ann in anns:
            question_id = ann['question_id']
            prompt = ann['image_object']
            dir = os.path.join(image_root, str(question_id))
            if os.path.exists(dir):
                continue
            os.makedirs(dir, exist_ok=True)
            index=0
            images = pipe(
                        prompt=prompt,
                        num_inference_steps=n_steps,
                        num_images_per_prompt=num_images_per_prompt
                    ).images
            
            # images = pipe(prompt, num_images_per_prompt=num_images_per_prompt//2).images
            for image in images:
                image.save(os.path.join(dir, f'{index}.jpg'), format='JPEG')
                index += 1
            print(f"Saved {num_images_per_prompt} images for question {question_id}")
    return

def main(args):
    pipe = load_diffusion_xl_model()
    anns = load_prompt(partition=args.partition, every=args.every)
    generate_images(anns, pipe)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=int, default=0)
    parser.add_argument('--every', type=int, default=253)
    args = parser.parse_args()
    main(args)
