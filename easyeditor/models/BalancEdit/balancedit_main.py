from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .balancedit import BalancEdit
from .balancedit_hparams import BalancEditHyperParams, BalancEditMultimodalHyperParams
from .utils import tokenize
from ...util import nethook


def apply_BalancEdit_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: BalancEditHyperParams,
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
    editor = BalancEdit(model=model, config=hparams,device=device)

    for request in requests:
        token,rephrase_token,locality_token = tokenize(request,tokenizer=tok,device=device, hparams=hparams)
        # print(token)
        editor.edit(config=hparams,tokens=token,rephrase_tokens=rephrase_token,locality_tokens=locality_token)
    with torch.no_grad():
        for w_name in hparams.inner_params:
            w_name=w_name.replace("[", ".").replace("]", "")
            w = nethook.get_parameter(editor.model,w_name)
            weights_copy[w_name]=w

    if not keep_original_weight:
        weights_copy = {}

    return editor,weights_copy


