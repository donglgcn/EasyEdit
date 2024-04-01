from dataclasses import dataclass
from typing import List, Optional
from ...util.hparams import HyperParams
import yaml


@dataclass
class BalancEditHyperParams(HyperParams):
    # Experiments
    
    edit_lr: int
    n_iter: int
    # Method
    eps: float
    dist_fn: str
    val_init: str
    val_train: str
    val_reg: str
    reg: str
    replacement: str
    eps_expand: str
    num_pert: str
    dropout: float

    # Module templates
    inner_params: List[str]
    device: int
    alg_name: str
    model_name: str

    # Defaults
    batch_size: int = 128
    max_length: int = 30
    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'BalancEdit') or print(
            f'BalancEditHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)

@dataclass
class BalancEditMultimodalHyperParams(HyperParams):
    # Method
    results_dir: str

    # Module templates
    device: int
    name: str
    alg_name: str
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    sentence_model_name: str

    # Experiments
    
    edit_lr: int
    n_iter: int
    # Method
    eps: float
    dist_fn: str
    val_init: str
    val_train: str
    val_reg: str
    reg: str
    replacement: str
    eps_expand: str
    num_pert: str
    dropout: float
    alpha: float

    # Module templates
    inner_params: List[str]

    ## Multimodal
    task_name: str
    qformer_checkpoint: str
    qformer_name_or_path: str
    state_dict_file: str
    
    # Image_dir
    coco_image: str
    rephrase_image: str  
    pretrained_ckpt: Optional[str] = None  

    # Defaults
    batch_size: int = 128
    max_length: int = 30
    model_parallel: bool = False
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'BalancEdit') or print(f'BalancEditMultimodalHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
