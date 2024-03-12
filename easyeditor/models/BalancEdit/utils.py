import transformers
import torch
import os
import numpy as np
import datetime
import struct
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def get_inner_params(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]

def param_subset(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [param_dict[n] for n in inner_names]

def parent_module(model, pname):
    components = pname.split('.')
    parent = model

    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")

    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")

    return parent

def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10**digits)

    return uuid.uuid_value

def ckpt_dir():
    """returns the directory in which to store model checkpoints"""
    path = "./ckpts/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def brackets_to_periods(name):
    return name.replace("[", ".").replace("]", "")
    
def get_params(model):
    return model.state_dict()

def get_shape(p, model): 
    # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
    return p.shape if isinstance(model, transformers.GPT2LMHeadModel) else (p.shape[1], p.shape[0])

def get_logits(x):
    return x.logits if hasattr(x, "logits") else x

def tokenize(requests, tokenizer, device, hparams, test=False):
    if not isinstance(requests, list):
        requests = [requests]
    src = [request["prompt"] for request in requests]
    trg = [request["target"] for request in requests]

    # trg = [
    #         (" " if request["target"][0] != " " else "")
    #         + request["target"]
    #         for request in requests
    #     ]
    image = [request["image"] for request in requests]
    image = torch.stack(image, dim=0).to(device)
    text_input = [s + " "+ t for s, t in zip(src, trg)]
    
    if hparams.model_name == "minigpt4":
        prompts_len = [len(tokenizer.encode(s, add_special_tokens=False)) for s in src]
        labels = tokenizer(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"].to(device)
    else:
        prompts_len = [len(tokenizer.encode(s)) for s in src]
        labels = tokenizer(trg, return_tensors="pt",)["input_ids"].to(device)

    # Run MEND
    tokens = dict(
        image=image,
        text_input=text_input,
        labels=labels,
        prompts_len=prompts_len
    )
    
    src_rephrase = [request["rephrased_questions_train"][0] for request in requests]
    # image_repharse = [request["image_rephrases"][0] for request in requests]
    # image_repharse = torch.stack(image_repharse, dim=0).to(device)
    text_input_repharse = [s + " "+ t for s, t in zip(src_rephrase, trg)]
    locality_answer = [request["locality_answer_train"] for request in requests]
    if hparams.model_name == "minigpt4":
        prompts_len_repharse = [len(tokenizer.encode(s, add_special_tokens=False)) for s in src_rephrase]
        labels_locality = tokenizer(locality_answer, add_special_tokens=False, return_tensors="pt",)["input_ids"].to(device)
    else:
        prompts_len_repharse = [len(tokenizer.encode(s)) for s in src_rephrase]
        labels_locality = tokenizer(locality_answer, return_tensors="pt",)["input_ids"].to(device)
    
    tokens_rephrase = dict(
        image=image, # image_repharse,
        text_input=text_input_repharse,
        labels=labels,
        prompts_len=prompts_len
    )
    text_input_locality = [s + " "+ t for s, t in zip(src, locality_answer)]
    tokens_locality = dict(
        image=image,
        text_input=text_input_locality,
        labels=labels_locality,
        prompts_len=prompts_len
    )
    # prompt, label = batch["prompt"], batch["target_new"]
    # if not isinstance(prompt, list):
    #     prompt=[prompt]
    # if not isinstance(label, list):
    #     label=[label]
    # mask_token = -100 # ignore_index of CrossEntropyLoss
    # if test or not label:
    #     tokens = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)
    #     tokens["labels"] = tokens["input_ids"].clone()
    #     tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token

    # else:
    #     full_prompt = [f"{p} {l}" for p, l in zip(prompt, label)]
    #     prompt_ids = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
    #     num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in prompt_ids]
    #     tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    #     tokens["labels"] = tokens["input_ids"].clone()
    #     for i in range(len(prompt)):
    #         tokens["labels"][i][:num_prompt_toks[i]] = mask_token

    #     tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
    
    # tokens = {f"{k1}" : v1.to(device) for k1, v1 in tokens.items()}
    return tokens, tokens_rephrase, tokens_locality

