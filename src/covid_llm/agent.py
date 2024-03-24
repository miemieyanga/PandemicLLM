import logging
import os
import types
from collections import OrderedDict
from collections import defaultdict
from collections.abc import Iterable

import deepspeed
import numpy as np
import torch as th
import torch.optim as optim
from omegaconf import OmegaConf

from utils.basics import init_path, lot_to_tol, time_logger
from utils.pkg.distributed import master_process_only
from .model import IGNORE_INDEX
from .metrics import calc_acc, calc_mse_from_cls_labels, calc_prediction_distribution,\
    calc_weighted_mse_from_cls_labels, calc_prediction_class_distribution, calc_brier_score

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Agent:

    def __init__(self, model, cfg, data, logger):
        self.cfg = cfg
        self.model = model
        self.data = data
        self.logger = logger
        self.optimizer = optim.Adam(model.parameters(), **cfg.ds.optimizer.params)
        self.total_batch_steps = cfg.total_steps * cfg.get('grad_acc_steps', 1)

    def forward(self, batch):
        return self.model(batch)

    def backward(self, loss):
        # Backward pass and optimization
        self.optimizer.zero_grad()  # Clear the gradients
        loss.backward()  # Calculate gradients
        self.optimizer.step()  # Update weights

    def torch_distributed_barrier(self):
        if self.cfg.get('world_size', 1) > 1:
            th.distributed.barrier()
            
    @time_logger()
    def evaluate_ICL(self, eval_iter, logger):
        output = {}
        for eval_batch in eval_iter:
            output = self.predict(eval_batch, self.cfg.eval_choice_only)
            for col in ['dialog', 'confidence', 'pred']:
                self.data.df.loc[list(eval_batch[0]), f'{col}_zero_shot'] = output[col]
                
        logger.info(f'Example generated output in zero_shot: {output["dialog"]}\n\n')    
        logger.info(f'Example generated output in zero_shot: {output["confidence"]}\n\n')    

    @time_logger()
    def evaluate(self, eval_iter_dict, logger, df_suffix=None):
        results = {}
        for split, eval_iter in eval_iter_dict.items():            
            eval_res = defaultdict(list)
            for eval_batch in eval_iter:
                output = self.predict(eval_batch, self.cfg.eval_choice_only)
                # Update dataframe to save
                for col in ['dialog', 'confidence', 'pred']:
                    if df_suffix is not None:
                        self.data.df.loc[list(eval_batch[0]), f'{col}_{df_suffix}'] = output[col]
                    else:
                        self.data.df.loc[list(eval_batch[0]), col] = output[col]
                for item, value in output.items():
                    eval_res[item].append(value)
            eval_res = {k: np.concatenate(v) if isinstance(v[0], Iterable) else np.array(v)
                        for k, v in eval_res.items()}
            results.update({f'{split}/{k}': np.array(eval_res[k]).mean()
                            for k in ['loss', 'token_acc'] if k in eval_res})
            logger.info(f'Example generated output in {split}: {eval_res["dialog"][:2]}\n\n\n')
            label, pred = eval_res['label'], eval_res['pred']
            if not self.cfg.add_class_token:
                results[f'{split}_valid_choice_rate'] = valid_choice_rate = np.mean(eval_res['is_valid'])
                if valid_choice_rate < 1:
                    logger.warning(f'Failed gold and prediction samples:\n'
                                   f"{list(zip(label[~eval_res['is_valid']], pred[~eval_res['is_valid']]))[:50]}")
            if 'acc' in self.cfg.metrics:
                results[f'{split}_acc'] = calc_acc(label, pred)
            if 'mse' in self.cfg.metrics:
                results[f'{split}_mse'] = calc_mse_from_cls_labels(label, pred, self.data.mse_val_map)
            if 'wmse' in self.cfg.metrics:
                results[f'{split}_wmse'] = calc_weighted_mse_from_cls_labels(label, eval_res['confidence'], self.data.mse_val_map)
            if 'bs' in self.cfg.metrics:
                results[f'{split}_bs'] = calc_brier_score(label, eval_res['confidence'], self.data.mse_val_map)
                
            pd_dict = calc_prediction_distribution(pred, self.model.cls_token_names)
            results.update({f'{split}-PD/{k}': v for k, v in pd_dict.items()})
            
        logger.warning(results)
        return results

    @th.no_grad()
    def predict(self, batch, choice_only=False):
        self.model.eval()
        node_ids, prompt_tree_lol, conversation_list = batch
        gold_text = [conv[1]['value'] for conv in conversation_list]
        inputs = {
            'batch': batch,
            'max_tgt_len': self.cfg.max_gen_len,
            'temperature': 0.2,
            'gold_text': gold_text,
        }
        batch_output = self.model.generate(inputs, choice_only)
        # print(batch_output) # For debug only
        batch_output['label'], is_valid = lot_to_tol([self.model.match_label_from_text(text) for text in gold_text])
        assert sum(is_valid) == len(is_valid), 'Incorrect gold text generation'
        # assert 'Not Found' not in label
        batch_output['pred'], batch_output['is_valid'] = lot_to_tol(
            [self.model.match_label_from_text(text) for text in batch_output['generated_text']])
        if 'loop_generated_text' in batch_output:
            batch_output['loop_pred'], _ = lot_to_tol(
                [self.model.match_label_from_text(text) for text in batch_output['loop_generated_text']])
        return batch_output

    def train_model_batch(self, batch, current_step=0):
        self.model.train()
        outputs, targets = self.forward(batch)  # Model forward

        loss = outputs.loss
        # calculate the token accuracy;
        # [B, S-1], BOS and one token shift next token prediction
        chosen_tokens = th.max(outputs.logits, dim=-1)[1][:, 1:-1]
        labels = targets[:, 2:]  # BOS + space
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(th.long)  # [B*S]
        valid_mask = (labels != IGNORE_INDEX).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item() if valid_mask.sum() > 0 else 0

        self.backward(loss)
        # self.progress.update(self.train_task, advance=1)
        return {'train/step': current_step, 'train/loss': round(loss.item(), 4), 'train/token_acc': round(gen_acc, 2), }

    @master_process_only
    def save_model(self, save_path, step, is_final=False):
        checkpoint_name = f"checkpoint_{step}" if not is_final else "final_model"
        # create save directory if not exists
        path = init_path(f"{save_path}{checkpoint_name}/")
        # only save trainable model parameters
        checkpoint = OrderedDict()
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                checkpoint[k] = v.detach().cpu()
        th.save(checkpoint, f'{path}/pytorch_model.pt')
        # save tokenizer
        self.model.tokenizer.save_pretrained(path)
        # save configuration
        self.model.llm.config.use_cache = True
        self.model.llm.config.save_pretrained(path)
        self.model.llm.config.use_cache = False
        self.torch_distributed_barrier()
        self.logger.info(f"Saved model into {path}")


class DeepSpeedAgent(Agent):

    def __init__(self, model, cfg, data, logger):
        super(DeepSpeedAgent, self).__init__(model, cfg, data, logger)
        # load config parameters of deepspeed
        ds_params = OmegaConf.to_object(cfg.ds)
        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**cfg)
        )
        self.model = self.ds_engine.module  # Overwrite with deepspeed module
        self.torch_distributed_barrier()

    def forward(self, batch):
        return self.ds_engine(batch)

    def backward(self, loss):
        self.ds_engine.backward(loss)
        self.ds_engine.step()
