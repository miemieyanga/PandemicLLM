import string

import hydra.utils
import torch as th
from tqdm import tqdm

th.set_num_threads(1)

from bidict import bidict
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedShuffleSplit

import utils.basics as uf
from .prompt_tree import PromptTree
from utils.basics import logger
import numpy as np
from functools import partial

CONTINUOUS_COLS = ['hospitalization_per_100k', 'reported_cases_per_100k', 'previous_infection_12w', 'Dose1_Pop_Pct',
                   'Series_Complete_Pop_Pct', 'Additional_Doses_Vax_Pct']


class CovidData:
    @uf.time_logger("dataset initialization")
    def __init__(self, cfg: DictConfig):  # Process split settings, e.g. -1/2 means first split
        self.cfg = cfg
        # ! Initialize Data Related
        self.raw_data = raw_data = uf.pickle_load(cfg.data.raw_data_file)
        self.df = df = raw_data.sta_dy_aug_data
        self.mse_val_map = raw_data.mse_val_map
        self.split_ids = splits = raw_data[cfg.splits_type][cfg.data.split]
        label_info = raw_data.label_info
        target_type = {'t': 'trend', 'r': 'risk'}[cfg.target[0]]
        self.label_info = label_info = label_info[label_info['label_type'] == target_type]
        logger.info(f'Loaded meta information of states')
        logger.info(f'Loaded COVID data, {len(df)} weeks in total')

        # ! Splits
        for split, split_id in splits.items():
            split_df = df.iloc[split_id]
            logger.info(
                f'{split.upper()} set ({len(split_df)}): from {split_df.Week_start.min()} to '
                f'{split_df.Week_start.max()}')

        # ! Get label names
        self.label_names = str(self.label_info.label_name.tolist())
        if self.cfg.remove_quotation:
            self.label_names = self.label_names.replace('"', "").replace("'", "")

        # ! Initialize Prompt Related
        # Initialize classification prompt
        assert (col := f"label_{cfg.data.label_text}") in self.label_info.columns, "Unknown classification prompt mode."
        cfg.data.label_description = "[" + ", ".join(
            f'{_.label_token}: {_[col]}' for i, _ in self.label_info.iterrows()) + "]"

        self.prompt = hydra.utils.instantiate(cfg.prompt)
        uf.logger.info(self.prompt.human)

        self.in_cont_fields = cfg.in_cont_fields
        # ! Initialize Sequential Data
        process_seq_col = lambda x, col: x[col][:cfg.in_weeks]
        for col in tqdm(cfg.data.dynamic_cols, 'Processing sequential data'):
            self.df[col] = df.apply(partial(process_seq_col, col=col), axis=1)
        return

    def __getitem__(self, item):
        return self.df.iloc[item]

    def build_demo_prompt(self, support_tree_list):
        if len(support_tree_list) > 0:
            demo_cfg = self.cfg.demo
            sep = '\n' * demo_cfg.n_separators
            demonstration = sep.join(
                self.prompt.demo_qa(info=t.prompt, answer=t.label) for t in support_tree_list)
            demo_prompt = self.prompt.demo(demonstration=demonstration)
            return demo_prompt
        else:
            return ''

    def build_prompt_tree(self, id, supervised=False, use_variant_prompt=False):
        label = self.df.iloc[id][self.cfg.target] if supervised else None
        prompt_tree = PromptTree(self.cfg, data=self, id=id,
                                 label=label, name_alias=self.cfg.tree_node_alias,
                                 style=self.cfg.prompt.style, use_variant_prompt=use_variant_prompt)
        return prompt_tree

    def select_demo(self, select_method, node_id):
        if (n_demos := self.cfg.demo.n_samples) <= 0:
            return []
        one_fixed_sample_for_each_class_funcs = ['first', 'max_degree']
        if select_method in one_fixed_sample_for_each_class_funcs:
            n_demo_per_class = max(n_demos // self.n_labels, 1)
            # Overwrite n_demos
            if select_method == 'first':  # Initialize if haven't
                demo_ids = np.concatenate(
                    [self.split_ids.train[np.where(self.labels[self.split_ids.train] == l)[0][:n_demo_per_class]] for l
                     in
                     np.arange(self.n_labels)])
            else:
                raise ValueError(f'Unsupported demo selection method {select_method}')
            return demo_ids
