from dataclasses import dataclass
from typing import List, Optional
import yaml

from ...util.hparams import HyperParams


@dataclass
class MMFTHyperParams(HyperParams):
    # Method
    results_dir: str
    # layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    # Module templates
    inner_params: List[str]
    device: int
    name: str
    alg_name: str
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    sentence_model_name: str

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
    batch_size: int = 64
    max_length: int = 40
    model_parallel: bool = False
    tune_rephrase: bool = False
    tune_locality: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'MMFT') or print(f'MMFTHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
