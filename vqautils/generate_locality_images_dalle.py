import torch
import json
import os
import os
import requests
from tqdm import tqdm
from openai import OpenAI, BadRequestError
from KEY import API_KEY
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def load_diffusion_model(model_id="stabilityai/stable-diffusion-2-1", device="cuda" if torch.cuda.is_available() else "cpu"):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32,)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.requires_safety_checker = False
    pipe.safety_checker = None
    pipe = pipe.to(device)
    return pipe

client = OpenAI(
    # This is the default and can be omitted
    api_key=API_KEY,
)
import json
from tqdm import tqdm
IMAGE_ROOT = '/localtmp/ktm8eh/datasets/VQA/locality_images_dalle2/'

# using stable diffusion to catch the exception when dalle fails to generate images
os.environ['HF_HOME'] = '/localtmp/ktm8eh/.cache/huggingface'

def load_prompt(prompt_file='./vqautils/mscoco_val2014_annotations.json', partition=0, every=505):
    with open(prompt_file, 'r') as file:
        prompt = json.load(file)
    anns = prompt['annotations'][partition*every: (partition+1)*every]
    return anns

def generate_image_object_dalle2(image_object, save_dir, num_images_per_prompt=10):
    # Call the OpenAI API with a prompt to generate multiple paraphrases in one go
    response = client.images.generate(
        model="dall-e-2",
        prompt="{}".format(image_object),
        size="256x256",
        # quality="standard",
        n=num_images_per_prompt,
    )

    image_urls = response.data
    for index, image_url in enumerate(image_urls):
        print(image_url.url)
        # download and save images
        img = requests.get(image_url.url).content
        with open(os.path.join(save_dir, f'{index}.png'),'wb') as f:
            f.write(img)
    return 

def generate_images_dalle2(anns, num_images_per_prompt=10, image_root=IMAGE_ROOT, exception=True):
    with torch.no_grad():
        if exception:
            pipe = load_diffusion_model()
        for ann in anns:
            question_id = ann['question_id']
            prompt = ann['locality_image_object']
            dir = os.path.join(image_root, str(question_id))
            if os.path.exists(dir):
                continue
            os.makedirs(dir, exist_ok=True)
            try:
                generate_image_object_dalle2(prompt, save_dir=dir, num_images_per_prompt=num_images_per_prompt)
            except BadRequestError:
                if not exception:
                    raise BadRequestError
                print(f"BadRequestError when generating images for question {question_id}")
                print("using stable diffusion as the alternative")
                # OOM error
                # images = pipe(prompt, num_images_per_prompt=num_images_per_prompt).images
                # for index, image in enumerate(images):
                #     image.save(os.path.join(dir, f'{index}.png'), format='png')
                for index in range(num_images_per_prompt):
                    image = pipe(prompt, num_images_per_prompt=1).images[0]
                    image.save(os.path.join(dir, f'{index}.png'), format='png')
            print(f"Saved {num_images_per_prompt} images for question {question_id}")
    return

def main(args):
    anns = load_prompt(partition=args.partition, every=args.every)
    generate_images_dalle2(anns, 1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=int, default=0)
    parser.add_argument('--every', type=int, default=5046)
    args = parser.parse_args()
    main(args)
