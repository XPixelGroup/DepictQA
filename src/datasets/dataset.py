import json
import os

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, root_dir, meta_paths_weights):
        super(TrainDataset, self).__init__()
        self.root_dir = root_dir
        meta_paths = [_[0] for _ in meta_paths_weights]
        meta_weights = [_[1] for _ in meta_paths_weights]
        self.metas = []
        for meta_weight, meta_path in zip(meta_weights, meta_paths):
            with open(os.path.join(root_dir, meta_path)) as fr:
                self.metas += meta_weight * json.load(fr)
        print(f"[!] Collect {len(self.metas)} samples for training")

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        if "quality" not in meta["task_type"]:
            img, img_A, img_B = meta["image"], None, None
        else:
            img, img_A, img_B = meta["image_ref"], meta["image_A"], meta["image_B"]
        img_path = os.path.join(self.root_dir, img) if img else None
        img_A_path = os.path.join(self.root_dir, img_A) if img_A else None
        img_B_path = os.path.join(self.root_dir, img_B) if img_B else None
        return {
            "img_path": img_path,
            "img_A_path": img_A_path,
            "img_B_path": img_B_path,
            "conversation": meta["conversations"],
            "task_type": meta["task_type"],
        }

    def collate(self, instances):
        res = dict()
        for key in instances[0].keys():
            res[key] = [instance[key] for instance in instances]
        return res


class ValDataset(Dataset):
    def __init__(self, root_dir, meta_path, dataset_name, task_name):
        super(ValDataset, self).__init__()
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.task_name = task_name
        with open(os.path.join(root_dir, meta_path)) as fr:
            self.metas = json.load(fr)
        print(f"[!] Collect {len(self.metas)} samples for inference")

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        img, img_A, img_B = meta["image_ref"], meta["image_A"], meta["image_B"]
        img_path = os.path.join(self.root_dir, img) if img else None
        img_A_path = os.path.join(self.root_dir, img_A) if img_A else None
        img_B_path = os.path.join(self.root_dir, img_B) if img_B else None
        return {
            "id": meta["id"],
            "img_path": img_path,
            "img_A_path": img_A_path,
            "img_B_path": img_B_path,
            "query": meta["query"],
        }

    def collate(self, instances):
        res = dict()
        for key in instances[0].keys():
            res[key] = [instance[key] for instance in instances]
        return res
