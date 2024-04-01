from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time

import torch
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)

LOG = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

    def edit_step(self, batch, training: bool, caption=False):
        self.model.train(training)
        self.original_model.train(training)

        with torch.no_grad():
            base_outputs = self.model(batch["loc"])
            if not isinstance(base_outputs, torch.Tensor):
                base_logits = base_outputs.logits
            else:  
                base_logits = base_outputs
                
            base_image_outputs = self.model(batch["loc_image"])
            if not isinstance(base_image_outputs, torch.Tensor):
                base_image_logits = base_image_outputs.logits
            else:
                base_image_logits = base_image_outputs
        
            if caption:
                pre_caption = self.model.generate(batch['caption'])
        # Do the edit

        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        edit_time = time.time() - start

        with torch.set_grad_enabled(training):
            if caption:
                post_caption = edited_model.generate(batch['caption'])
                print("pre_caption: ", pre_caption)
                print("post_caption: ", post_caption)
            # Editing loss
            # print(batch["edit_outer"]['text_input'])
            if isinstance(batch["edit_outer"]['text_input'][0], list):
                # convert 10 text_input to 10 batches of text_input with original batch size
                post_edit_logitss = []
                post_batch_labelss = []
                new_batch_edit_outer = []
                b = batch["edit_outer"]
                # print("b_len: ", len(b["text_input"][0]))
                for i in range(len(b["text_input"][0])):
                    new_b = copy.deepcopy(b)
                    new_b["text_input"] = [t[i] for t in b["text_input"]]
                    # print("new_b: ", new_b["text_input"])
                    new_b['prompts_len'] = [p[i] for p in b["prompts_len"]]
                    new_batch_edit_outer.append(new_b)
                for i, b in enumerate(new_batch_edit_outer):
                    post_edit_output = edited_model(b)
                    post_batch_label = batch["edit_outer"]["labels"]
                    # print("post_batch_label: ", post_batch_label)
                    if not isinstance(post_edit_output, torch.Tensor):
                        post_edit_logit = post_edit_output.logits
                    else:
                        post_edit_logit = post_edit_output
                    post_edit_logitss.append(post_edit_logit)
                    post_batch_labelss.append(post_batch_label)
                    # print("post_edit_logit: ", post_edit_logit.shape)
                    # print("post_batch_label: ", post_batch_label.shape)
                # post_edit_logits = torch.cat(post_edit_logits, dim=0)
                # print("post_edit_logits: ", post_edit_logits.shape)
                # post_batch_labels = torch.cat(post_batch_labels, dim=0)
                # print("post_batch_labels: ", post_batch_labels.shape)
            else:
                post_edit_outputs = edited_model(batch["edit_outer"])
                post_batch_labelss = [batch["edit_outer"]["labels"]]
                if not isinstance(post_edit_outputs, torch.Tensor):
                    post_edit_logitss = [post_edit_outputs.logits]
                else:
                    post_edit_logitss = [post_edit_outputs]
            

            # rephrase image
            if batch["edit_outer_image"]['image'].dim() == 5:
                post_image_edit_logitss = []
                post_image_batch_labelss = []
                new_batch_edit_outer_image = []
                b = batch["edit_outer_image"]
                for i in range(len(b["image"][0])):
                    new_b = copy.deepcopy(b)
                    new_b["image"] = b['image'][:, i, :, :, :]
                    new_batch_edit_outer_image.append(new_b)
                for i, b in enumerate(new_batch_edit_outer_image):
                    post_image_edit_output = edited_model(b)
                    post_image_batch_label = batch["edit_outer_image"]["labels"]
                    if not isinstance(post_image_edit_output, torch.Tensor):
                        post_image_edit_logit = post_image_edit_output.logits
                    else:
                        post_image_edit_logit = post_image_edit_output
                    post_image_edit_logitss.append(post_image_edit_logit)
                    # print("post_image_edit_logit: ", post_image_edit_logit.shape)
                    post_image_batch_labelss.append(post_image_batch_label)
                    # print("post_image_batch_label: ", post_image_batch_label.shape)
                # post_image_edit_logits = torch.cat(post_image_edit_logits, dim=0)
                # print("post_image_edit_logits: ", post_image_edit_logits.shape)
                # post_image_batch_labels = torch.cat(post_image_batch_labels, dim=0)
                # print("post_image_batch_labels: ", post_image_batch_labels.shape)
            else:
                post_image_edit_outputs = edited_model(batch["edit_outer_image"])
                post_image_batch_labelss = [batch["edit_outer_image"]["labels"]]
                if not isinstance(post_image_edit_outputs, torch.Tensor):
                    post_image_edit_logitss = [post_image_edit_outputs.logits]
                else:
                    post_image_edit_logitss = [post_image_edit_outputs]
                
            inner_edit_outputs = edited_model(batch["edit_inner"])
            inner_batch_labels = batch["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            l_edits = 0.0
            for post_edit_logits, post_batch_labels in zip(post_edit_logitss, post_batch_labelss):
                if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                    l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)["nll"]
                else:
                    l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])["nll"]
                l_edits += l_edit
            l_edit = l_edits / len(post_edit_logitss)

            l_image_edits = 0.0
            for post_image_edit_logits, post_image_batch_labels in zip(post_image_edit_logitss, post_image_batch_labelss):
                if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                    l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)["nll"]
                else:
                    l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])["nll"]  
                l_image_edits += l_image_edit
            l_image_edit = l_image_edits / len(post_image_edit_logitss)             
            
            # Collect some useful metrics
            with torch.no_grad():
                post_edit_dicts = []
                for post_edit_logits, post_batch_labels in zip(post_edit_logitss, post_batch_labelss):
                    if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                        post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
                    else:
                        post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
                    post_edit_dicts.append(post_edit_dict)
                post_edit_dict = {k: sum([d[k] for d in post_edit_dicts])/len(post_edit_logitss) for k in post_edit_dicts[0].keys()}
                
                if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
                else:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
                print("inner_edit_logits: ", inner_edit_logits.shape)
                print("inner_batch_labels: ", inner_batch_labels.shape)
                print("inner_edit_logits: ", inner_edit_logits.argmax(-1)[:10])
                print("inner_batch_labels: ", inner_batch_labels)

                image_rephrase_edit_dicts = []
                for post_image_edit_logits, post_image_batch_labels in zip(post_image_edit_logitss, post_image_batch_labelss):
                    if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                        image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
                    else:
                        image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
                    image_rephrase_edit_dicts.append(image_rephrase_edit_dict)
                image_rephrase_edit_dict = {k: sum([d[k] for d in image_rephrase_edit_dicts])/len(post_image_edit_logitss) for k in image_rephrase_edit_dicts[0].keys()}
            
            post_base_outputs = edited_model(batch["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
                kl_mask = post_base_outputs.attention_mask
            else:
                post_base_logits = post_base_outputs
                kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(post_base_logits.device)

            post_image_base_outputs = edited_model(batch["loc_image"])
            post_local_batch_labels = batch["loc_image"]["labels"]
            # print("post_local_batch_labels: ", post_local_batch_labels)
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
                kl_image_mask = post_image_base_outputs.attention_mask
            else:
                post_image_base_logits = post_image_base_outputs
                kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(base_image_logits.device)
            # print("post_image_base_logits: ", post_image_base_logits.shape)
            # print("post tokens: ", post_image_base_logits.argmax(-1)[0])
            # print("pre tokens: ", base_image_logits.argmax(-1)[0])
            base_image_outputs_len = (base_image_outputs.labels == -100).sum(dim=-1)-2 # doesn't count bos token and index starts from 0
            inner_edit_outputs_len = (inner_edit_outputs.labels == -100).sum(dim=-1)-2 

            truncated_base_image_logits = base_image_logits[:, base_image_outputs_len[0]:]
            truncated_post_image_base_logits = post_image_base_logits[:, base_image_outputs_len[0]:]
            # for i, index in enumerate(base_image_outputs_len):
            #     truncated_base_image_logits[i] = base_image_logits[i, index:, :]
            #     truncated_post_image_base_logits[i] = post_image_base_logits[i, index:, :]
            # print("['base_image_outputs_len']: ", base_image_outputs_len, base_image_outputs.labels)
            # print("['inner_edit_outputs_len']: ", inner_edit_outputs_len, inner_edit_outputs.labels)
            # print("post_local_batch_labels: ", post_local_batch_labels[0], post_image_base_outputs.labels)

            # print("truncated_base_image_logits: ", truncated_base_image_logits)
            # print("truncated_post_image_base_logits: ", truncated_post_image_base_logits)
            print("base output: ", self.val_set.tok.decode(truncated_base_image_logits.argmax(-1)[0]))
            print("post output: ", self.val_set.tok.decode(truncated_post_image_base_logits.argmax(-1)[0]))

            # Collect some useful metrics for locality accuracy
            with torch.no_grad():
                if base_image_logits.shape[1] > post_local_batch_labels.shape[1]:
                    pre_image_local_dict = self.model.edit_loss_fn(self.config, base_image_logits, post_local_batch_labels)
                else:
                    pre_image_local_dict = self.model.edit_loss_fn(self.config, base_image_logits, post_local_batch_labels[:, -post_base_logits.shape[1]-1:])
                if post_image_base_logits.shape[1] > post_local_batch_labels.shape[1]:
                    post_image_local_dict = self.model.edit_loss_fn(self.config, post_image_base_logits, post_local_batch_labels)
                else:
                    post_image_local_dict = self.model.edit_loss_fn(self.config, post_image_base_logits, post_local_batch_labels[:, -post_image_base_logits.shape[1]-1:])

            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)
            l_image_loc = kl_loc_loss(base_image_logits.detach(), post_image_base_logits, mask=kl_image_mask)

        if l_edit.isnan():
            print("l_edit is nan")
            print("input: ", batch["edit_outer"]['text_input'])
        elif l_image_edit.isnan():
            print("l_image_edit is nan")
            print("input: ", batch["edit_outer_image"]['text_input'])
        elif l_loc.isnan():
            print("l_loc is nan")
            print("input: ", batch["loc"]['text_input'])
        elif l_image_loc.isnan():
            print("l_image_loc is nan")
            print("input: ", batch["loc_image"]['text_input'])

        if self.config.alg == "SERAC_MULTI":
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc + self.config.iedit * l_image_edit
        else:
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_loc+l_image_loc) + self.config.iedit * l_image_edit
        

        if training and self.config.alg != 'ft':
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

        # Text locality
        post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
        base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

        # Image locality
        # revised: only use the target output only
        post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(truncated_post_image_base_logits, dim=-1), k=10, dim=-1).indices
        base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(truncated_base_image_logits, dim=-1), k=10, dim=-1).indices
        # print("post_image_base_logits_softmax_top_k: ", post_image_base_logits_softmax_top_k.shape)
        # print("post_image_base_logits_softmax_top_1: ", post_image_base_logits_softmax_top_k[0])
        # print("base_image_logits_softmax_top_1: ", base_image_logits_softmax_top_k[0])

        info_dict = {}
        info_dict['loss/edit'] = l_edit.item()
        info_dict['loss/image_edit'] = l_image_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict['edit/log_prob'] = post_edit_dict["log_prob"].item()
        info_dict['edit/prob'] = post_edit_dict["prob"].item()
        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        # add post_image_local_dict
        info_dict['image_loc/post_acc'] = post_image_local_dict["acc"].item()
        info_dict['image_loc/pre_acc'] = pre_image_local_dict["acc"].item()
        
        info_dict["time/edit"] = edit_time
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        image_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"
        loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        loc_image_acc = f"{stats['image_loc/acc_val']:<12.5f}"

        LOG.info(
          f"Step {prog} outer_acc: {outer_acc} image_acc: {image_acc} inner_acc: {inner_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}, image_loc: {loc_image_acc}"
        )

    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in enumerate(self.val_loader):
            if val_step >= steps:
                break
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict)

            if (
                log
                and (val_step + 1) % self.config.log_interval == 0
            ):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats