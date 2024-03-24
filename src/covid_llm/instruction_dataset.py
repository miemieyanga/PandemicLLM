from functools import partial

import torch as th
from torch.utils.data import Dataset, DataLoader, Subset

from utils.data.covid_data import CovidData
from .samplers import DistributedBatchSampler


def load_sft_dataset(cfg, full_dataset, split, split_ids, batch_size, world_size=1, rank=0):
    dataset = Subset(full_dataset, split_ids)
    if split == "train":
        sampler = th.utils.data.RandomSampler(dataset)
    else:
        sampler = th.utils.data.SequentialSampler(dataset)
    if split == "train" and world_size > 1:
        batch_sampler = DistributedBatchSampler(
            sampler, batch_size, True, rank, world_size
        )
        iter_ = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=partial(full_dataset.collate),
            pin_memory=True,
        )
    else:
        iter_ = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=partial(full_dataset.collate),
            pin_memory=True,
        )
    return dataset, iter_, sampler


class InstructionDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data: CovidData, cfg, mode, use_variant_prompt=False):
        super(InstructionDataset, self).__init__()
        self.data = data
        self.cfg = cfg
        self.mode = mode
        self.use_variant_prompt = use_variant_prompt

    def __len__(self):  # number of instances
        return len(self.data.df)

    def __getitem__(self, id):
        # ! Build Graph Trees
        support_tree_list = []  # For demonstrations in ICL
        if self.cfg.use_demo:
            demo_center_nodes = self.data.select_demo(self.cfg.demo.select_method, id)
            support_tree_list = [  # No node drop out for demo nodes
                self.data.build_prompt_tree(center_node, supervised=True)
                for center_node in demo_center_nodes]
        query_tree = self.data.build_prompt_tree(id, supervised=False, use_variant_prompt=self.use_variant_prompt)
        prompt_tree_list = support_tree_list + [query_tree]

        # ! Build Prompt
        demo = self.data.build_demo_prompt(support_tree_list)
        question = self.data.prompt.question(info=query_tree.prompt)
        in_text = self.data.prompt.human(demo=demo, question=question)
        if self.mode == 'sft':
            out_text = self.data.prompt.gpt(answer=self.data[id][self.cfg.target])
        else:
            out_text = None

        conversation = [
            {"from": "human", "value": in_text},
            {"from": "gpt", "value": out_text},
        ]

        return int(id), prompt_tree_list, in_text, out_text, demo, question, conversation

    def collate(self, batch):
        # Key: field,  Value: The list of continuous sequence to encode
        node_ids, prompt_tree_lol, in_text_list, out_text_list, demo_list, question_list, conversation_list = zip(
            *batch)
        return list(node_ids), prompt_tree_lol, conversation_list
