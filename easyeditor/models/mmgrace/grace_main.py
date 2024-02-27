from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .mmGRACE import MMGRACE
from .grace_hparams import MMGraceHyperParams, MMGraceMultimodalHyperParams
from .utils import tokenize
from ...util import nethook


def apply_mmgrace_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: MMGraceHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=True,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    # request = requests[0]
    if copy:
        model = deepcopy(model)
    weights_copy = {}
    device = torch.device(f'cuda:{hparams.device}')
    editor = MMGRACE(model=model, config=hparams,device=device)

    for request in requests:
        token = tokenize(request,tokenizer=tok,device=device, hparams=hparams)
        # print(token)
        editor.edit(config=hparams,tokens=token)
    # for request in requests:
    #     print(
    #         f"Executing GRACE algo for: "
    #         f"[{request['prompt']}] -> [{request['target_new']}]"
    #     )
    #     tokens = tokenize(request,tokenizer=tok,device=device)
    #     editor.edit(config=hparams,tokens=tokens)
    with torch.no_grad():
        for w_name in hparams.inner_params:
            w_name=w_name.replace("[", ".").replace("]", "")
            w = nethook.get_parameter(editor.model,w_name)
            weights_copy[w_name]=w

    if not keep_original_weight:
        weights_copy = {}

    return editor,weights_copy


