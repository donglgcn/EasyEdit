from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .mmft_hparams import MMFTHyperParams

class MMFT(torch.nn.Module):
    def __init__(self, model, deltas: Dict[str, torch.Tensor] = None):
        super(MMFT, self).__init__()
        self.model = model
        self.deltas = deltas
        self.edited = True
    
    def __call__(self, token):
        return self.model(token)
    
    def reset_layers(self):
        with torch.no_grad():
            for w_name, upd_matrix in self.deltas.items():
                w = nethook.get_parameter(self.model, w_name)
                w[...] -= upd_matrix
        self.edited = False
        return self
        
    def resume_layers(self):
        with torch.no_grad():
            for w_name, upd_matrix in self.deltas.items():
                w = nethook.get_parameter(self.model, w_name)
                w[...] += upd_matrix
        self.edited = True
        return self

def apply_mmft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MMFTHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    device = torch.device(f'cuda:{hparams.device}')
    deltas = execute_mmft(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    MMFT_model = MMFT(model, deltas)

    if not keep_original_weight:
        weights_copy = {}

    return MMFT_model, weights_copy


def execute_mmft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MMFTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    model = model.to(device)
    # Retrieve weights that user desires to change
    layers = hparams.inner_params
    # suffixes = [".weight", ".bias"]
    # layers = [layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer for layer in layers]
    # layer_names = [layer.rsplit(".", 1)[-1] for layer in layers]
    for n, p in model.named_parameters():
        p.requires_grad = False
    
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer_name in layers
        if layer_name in n
    }
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
        eps=1e-4,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    
    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        token,rephrase_token,locality_token = tokenize(request,tokenizer=tok,device=device, hparams=hparams)

        # if request["target_new"] != " ":
        #     # Space required for correct tokenization
        #     request["target_new"] = " " + request["target_new"]
        # print(
        #     f"Executing FT algo for: "
        #     f"[{request['prompt']}] -> [{request['target_new']}]"
        # )
    
        # # Define inputs
        # texts = [r["prompt"] for r in requests]
        # targets = [r["target_new"] for r in requests]
    
    
        # Update loop: intervene at layers simultaneously
        loss_meter = AverageMeter()
        for it in range(hparams.num_steps):
            print(20 * "=")
            print(f"Epoch: {it}")
            print(20 * "=")
            loss_meter.reset()
            opt.zero_grad()
            # bs = inputs["input_ids"].shape[0]
            # tokens = [token, rephrase_token, locality_token]
            tokens = [token]
            if hparams.tune_rephrase:
                tokens.append(rephrase_token)
            if hparams.tune_locality:
                tokens.append(locality_token)
            for token in tokens:
                if hparams.model_name == "minigpt4" or hparams.model_name == "blip2":
                    outputs = model(token)
                    loss = outputs.loss
                    # print("mend loss:" ,loss, "loss:", outputs.loss)
                    loss.backward()
                    # print("weights:", list(weights.values())[0], list(weights.values())[0].grad)
                    opt.step()
                print(f"loss {loss.item()}")
                loss_meter.update(loss.item())

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
    
    if hparams.model_name == "minigpt4" or hparams.model_name == "blip2":
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
        prompts_len=prompts_len_repharse
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
