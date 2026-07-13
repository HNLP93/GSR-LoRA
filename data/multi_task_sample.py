from collections import OrderedDict
import abc
import datasets
import functools
import logging
import numpy as np
import torch
from typing import Callable, Dict, Mapping, List
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import torch.distributed as dist
import sys
import random
from rank import *
logger = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = 'data/glue'

def _get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

class AbstractTaskDataset(abc.ABC):
    root_path = DEFAULT_DATA_ROOT
    name = NotImplemented
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}

    def __init__(self, seed=42, data_root=None, use_half_validation=False):
        self.seed = seed
        self.root_path = data_root or self.root_path
        self.use_half_validation = use_half_validation
            
    def load_dataset(self, split: int):
        dataset = load_from_disk(f'{self.root_path}/{self.name}_with_prompt')[split]
        return dataset

    def get_dataset(self, split):
        split = self.split_to_data_split[split]
        dataset = self.load_dataset(split=split)
        if self.use_half_validation and self.name in ['mrpc', 'rte', 'cola', 'stsb'] and split != 'train':
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            validation_size = len(dataset)
            indices = torch.randperm(validation_size, generator=generator).tolist()
            return dataset.select(indices[validation_size // 2:])
        return dataset



class MRPCTaskDataset(AbstractTaskDataset):
    name = "mrpc"
    label_list = ["0", "1"]
    target_map = {'0':'yes', '1':'no'}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}



class COLATaskDataset(AbstractTaskDataset):
    name = "cola"
    label_list = ["0", "1"]
    target_map = {'0':'no', '1':'yes'}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}



class SST2TaskDataset(AbstractTaskDataset):
    name = "sst2"
    label_list = ["0", "1"]
    target_map = {'0':'negative', '1':'positive'}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


class STSBTaskDataset(AbstractTaskDataset):
    name = "stsb"
    label_list = ['0', '1', '2', '3', '4', '5']
    target_map = {'0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5'}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


class QQPTaskDataset(AbstractTaskDataset):
    name = "qqp"
    label_list = ["0", "1"]
    target_map = {'0':'no', '1':'yes'}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


class MNLITaskDataset(AbstractTaskDataset):
    name = "mnli"
    label_list = ["0", "1", "2"]
    target_map = {'0':'positive', '1':'neutral', '2':'negative'}
    split_to_data_split = {"train": "train",
                           "validation": "validation_matched",
                           "test": "validation_mismatched"}

class QNLITaskDataset(AbstractTaskDataset):
    name = "qnli"
    label_list = ["0", "1"]
    target_map = {'0':'yes', '1':'no'}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


class RTETaskDataset(AbstractTaskDataset):
    name = "rte"
    label_list = ["0", "1"]
    target_map = {'0':'yes', '1':'no'}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


class MedMCQATaskDataset(AbstractTaskDataset):
    name = "medmcqa"
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}


class MagicoderTaskDataset(AbstractTaskDataset):
    name = "magicoder"
    split_to_data_split = {"train": "train", "validation": "validation", "test": "validation"}


class FinanceAlpacaTaskDataset(AbstractTaskDataset):
    name = "finance_alpaca"
    split_to_data_split = {"train": "train", "validation": "validation", "test": "validation"}


class MetaMathQATaskDataset(AbstractTaskDataset):
    name = "metamathqa"
    split_to_data_split = {"train": "train", "validation": "validation", "test": "validation"}


class AlpacaGPT4TaskDataset(AbstractTaskDataset):
    name = "alpaca_gpt4"
    split_to_data_split = {"train": "train", "validation": "validation", "test": "validation"}


class E2ENLGTaskDataset(AbstractTaskDataset):
    name = "e2e_nlg"
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}


class HumanEvalTaskDataset(AbstractTaskDataset):
    name = "humaneval"
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}


class GSM8KTaskDataset(AbstractTaskDataset):
    name = "gsm8k"
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}


class PhraseBankTaskDataset(AbstractTaskDataset):
    name = "phrasebank"
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}


class ARCCChallengeTaskDataset(AbstractTaskDataset):
    name = "arc_c"
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}


class ARCEasyTaskDataset(AbstractTaskDataset):
    name = "arc_e"
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}



TASK_MAPPING = OrderedDict([
    ('cola', COLATaskDataset),
    ('sst2', SST2TaskDataset),
    ('stsb', STSBTaskDataset),
    ('qqp', QQPTaskDataset),
    ('mnli', MNLITaskDataset),
    ('qnli', QNLITaskDataset),
    ('rte', RTETaskDataset),
    ('mrpc', MRPCTaskDataset),
    ('medmcqa', MedMCQATaskDataset),
    ('magicoder', MagicoderTaskDataset),
    ('finance_alpaca', FinanceAlpacaTaskDataset),
    ('metamathqa', MetaMathQATaskDataset),
    ('alpaca_gpt4', AlpacaGPT4TaskDataset),
    ('e2e_nlg', E2ENLGTaskDataset),
    ('humaneval', HumanEvalTaskDataset),
    ('gsm8k', GSM8KTaskDataset),
    ('phrasebank', PhraseBankTaskDataset),
    ('arc_c', ARCCChallengeTaskDataset),
    ('arc_e', ARCEasyTaskDataset)]
)


class AutoTask:
    @classmethod
    def get(self, task_name, seed=42, data_root=None, use_half_validation=False):
        if task_name in TASK_MAPPING:
            return TASK_MAPPING[task_name](
                seed=seed,
                data_root=data_root,
                use_half_validation=use_half_validation,
            )
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
        
class TaskCollator:
    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.max_target_len = self.calc_target_max_len()
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        input_batch = self.input_encode(batch)
        target_batch = self.target_encode(batch)
        decoder_input_ids = target_batch['input_ids'].clone()
        decoder_input_ids[:, 1:] = target_batch['input_ids'][:, :-1]
        decoder_input_ids[:, 0] = self.pad_token_id
        labels = target_batch['input_ids'].clone()
        labels[target_batch['input_ids'] == self.pad_token_id] = -100
        return {
            "input_ids":input_batch['input_ids'],
            "attention_mask":input_batch['attention_mask'],
            "decoder_input_ids":decoder_input_ids,
            "labels":labels
        }

    def input_encode(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer(
            [x["prompt"] for x in batch],
            padding='max_length',   
            return_tensors='pt',
            truncation=True, 
            max_length=self.data_args.max_length
        )
        return batch_encoding
    
    def target_encode(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer(
            [x["label"] for x in batch],
            padding='max_length',   
            return_tensors='pt',
            truncation=True, 
            max_length=self.max_target_len
        )
        return batch_encoding
    
    def calc_target_max_len(self):
        word_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
        word_list += ['unacceptable', 'acceptable', 'entailment', 'neutral', 'contradiction', 
                      'not_equivalent', 'equivalent', 'not_entailment', 'not_duplicate', 'duplicate', 'negative', 'positive']
        max_len = 0
        for word in word_list:
            ids = self.tokenizer.encode(word)
            max_len = max(max_len, len(ids))
            #print(word, ids, len(ids), self.tokenizer.decode(ids))
        return max_len
    
from datasets import Dataset
class MultiTaskConcateDataLoader(DataLoader):
    def __init__(self, datasets, data_collator, batch_size, data_args=None, seed=2023, **kwargs):
        self.datasets = datasets
        self.seed = seed
        random.seed(self.seed)
        self.data_args = data_args
        self.dataset_iters = [iter(loader) for loader in datasets]
        self.data_collator = data_collator
        self.data_sizes = [len(dataset) for dataset in self.datasets]
        self.batch_size = batch_size
        epoch_value = (
            getattr(self.data_args, "dataloader_epochs", None)
            or getattr(self.data_args, "epochs", None)
            or 1
        )
        self.num_epochs = max(1, int(np.ceil(float(epoch_value))))
        self.num_gpus = _get_world_size()
        self.total_steps = max(1, int((sum(self.data_sizes) * self.num_epochs) // (self.batch_size * self.num_gpus)))
        self.current_step = 0
        self.all_batches = []
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset = dataset.shuffle()
            dataset_iter = iter(dataset)  # 获取数据集的迭代器
            dataset_batches = []
            while True:
                batch = []
                for _ in range(self.batch_size):
                    sample = next(dataset_iter, None)
                    if sample is not None:
                        batch.append(sample)
                    else:
                        break  # 如果数据集中的样本已经用完，则退出内层循环
                if batch:  # 只有在批次中有样本时才添加到结果中
                    dataset_batches.append(batch)
                else:
                    break  # 如果数据集中的样本已经用完，则退出外层循环
            dataset_batches = [(batch, dataset_idx) for batch in dataset_batches]
            self.all_batches.extend(dataset_batches)
        random.shuffle(self.all_batches)
        super(MultiTaskConcateDataLoader, self).__init__(self.datasets, batch_size=batch_size, **kwargs)

    def __iter__(self):
        batch_iter = iter(self.all_batches)
        for i in range(self.num_epochs):
            while True:
                try:
                    batch, dataset_idx = next(batch_iter)
                    collated_sample = self.data_collator(batch)
                    set_task_id(dataset_idx)  # 设置当前batch所属数据集id
                    yield collated_sample
                except StopIteration:
                    self.all_batches = []
                    for dataset_idx, dataset in enumerate(self.datasets):
                        dataset = dataset.shuffle()
                        dataset_iter = iter(dataset)  # 获取数据集的迭代器
                        dataset_batches = []
                        while True:
                            batch = []
                            for _ in range(self.batch_size):
                                sample = next(dataset_iter, None)
                                if sample is not None:
                                    batch.append(sample)
                                else:
                                    break  # 如果数据集中的样本已经用完，则退出内层循环
                            if batch:  # 只有在批次中有样本时才添加到结果中
                                dataset_batches.append(batch)
                            else:
                                break  # 如果数据集中的样本已经用完，则退出外层循环
                        dataset_batches = [(batch, dataset_idx) for batch in dataset_batches]
                        self.all_batches.extend(dataset_batches)
                    random.shuffle(self.all_batches)
                    batch_iter = iter(self.all_batches)
                    break

    
    def __len__(self):
        # 返回数据集的长度
        return len(self.all_batches) * self.num_epochs // self.num_gpus
    
class MultiTaskDataLoader(DataLoader):
    def __init__(self, datasets, data_collator, batch_size, data_args=None, seed=2023, **kwargs):
        self.datasets = datasets
        self.seed = seed
        random.seed(self.seed)
        # for i, dataset in enumerate(self.datasets):
        #     self.datasets[i] = dataset.shuffle()
        self.data_args = data_args
        self.dataset_iters = [iter(loader) for loader in datasets]
        self.data_collator = data_collator
        self.data_sizes = [len(dataset) for dataset in self.datasets]
        self.dataset_probabilities = np.array(self.data_sizes) / sum(self.data_sizes)
        self.dataset_probabilities = np.exp(self.dataset_probabilities) / np.sum(np.exp(self.dataset_probabilities))
        #self.dataset_probabilities = np.array([1/7] * 7)
        #prob = [0.03, 0.75, 0.01, 0.25, 0.75, 0.01, 0.2]
        #prob = [0.1, 0.6, 0.05, 0.2, 0.6, 0.05, 0.15]
        #prob = [0.1, 0.5, 0.05, 0.2, 0.5, 0.05, 0.15]
        #prob = [0.15, 0.5, 0.1, 0.25, 0.5, 0.15, 0.2]
        #self.dataset_probabilities = np.array(prob) / sum(np.array(prob))
        print(self.dataset_probabilities)
        self.batch_size = batch_size
        self.num_epochs = self._resolve_num_epochs(self.data_args)
        num_gpus = _get_world_size()
        self.total_steps = max(1, int((sum(self.data_sizes) * self.num_epochs) // (self.batch_size * num_gpus)))
        self.current_step = 0
        super(MultiTaskDataLoader, self).__init__(datasets, batch_size=batch_size, **kwargs)

    @staticmethod
    def _resolve_num_epochs(data_args):
        if data_args is None:
            return 1.0
        dataloader_epochs = getattr(data_args, "dataloader_epochs", None)
        if dataloader_epochs is not None:
            return float(dataloader_epochs)
        legacy_epochs = getattr(data_args, "epochs", None)
        if legacy_epochs is not None:
            return float(legacy_epochs)
        return 1.0

    def __iter__(self):
        self.current_step = 0
        self.dataset_iters = [iter(dataset) for dataset in self.datasets]
        while self.current_step < self.total_steps:
            random_dataset_idx = np.random.choice(
                len(self.datasets),
                p=self.dataset_probabilities
            )
            set_task_id(random_dataset_idx)
            try:
                samples = [next(self.dataset_iters[random_dataset_idx]) for _ in range(self.batch_size)]
                collated_sample = self.data_collator(samples)
                yield collated_sample
                
            except StopIteration:
                #self.datasets[random_dataset_idx] = self.datasets[random_dataset_idx].shuffle()
                self.dataset_iters[random_dataset_idx] = iter(self.datasets[random_dataset_idx])
                samples = [next(self.dataset_iters[random_dataset_idx]) for _ in range(self.batch_size)]
                collated_sample = self.data_collator(samples)
                yield collated_sample

            self.current_step += 1
    
    def __len__(self):
        # 返回数据集的长度
        return self.total_steps
