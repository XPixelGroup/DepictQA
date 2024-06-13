import torch
from torch.utils.data import DataLoader

from .dataset import TrainDataset, ValDataset
from .samplers import DistributedBatchSampler


def load_trainset(args):
    dataset = TrainDataset(args.data["root_dir"], args.data.train["meta_paths_weights"])
    sampler = torch.utils.data.RandomSampler(dataset)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = (
        args["world_size"] * args.dschf.config["train_micro_batch_size_per_gpu"]
    )
    batch_sampler = DistributedBatchSampler(sampler, batch_size, True, rank, world_size)
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=args.data["num_workers"],
        collate_fn=dataset.collate,
        pin_memory=True,
    )
    return dataloader


def load_valset(args):
    dataset = ValDataset(
        args.data["root_dir"],
        args.data.infer["meta_path"],
        args.data.infer["dataset_name"],
        args.data.infer["task_name"],
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.data.infer["batch_size"],
        shuffle=False,
        num_workers=args.data["num_workers"],
        drop_last=False,
        collate_fn=dataset.collate,
        pin_memory=True,
    )
    return dataloader
