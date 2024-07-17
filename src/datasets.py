import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.signal import resample

def preprocess_data(X, new_sampling_rate=200, original_sampling_rate=1000):
    # データ型をfloat64に変換
    X = X.astype(np.float64)
    
    # リサンプリング
    num_samples = int(X.shape[-1] * new_sampling_rate / original_sampling_rate)
    X = resample(X, num_samples, axis=-1)
    
    # スケーリング
    X_mean = np.mean(X, axis=-1, keepdims=True)
    X_std = np.std(X, axis=-1, keepdims=True)
    X_std[X_std == 0] = 1  # 標準偏差がゼロの箇所を1に置き換え
    X = (X - X_mean) / X_std
    
    # ベースライン補正
    baseline = X[..., :int(0.2 * new_sampling_rate)]  # 例: 最初の200msをベースラインとして使用
    baseline_mean = np.mean(baseline, axis=-1, keepdims=True)
    X = X - baseline_mean
    
    return X


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", preprocess: bool = True, original_sampling_rate=1000) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.preprocess = preprocess
        self.original_sampling_rate = original_sampling_rate
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i].numpy()  # .ptファイルからロードされたテンソルをnumpy配列に変換
        if self.preprocess:
            X = preprocess_data(X, original_sampling_rate=self.original_sampling_rate)  # 前処理を適用
        X = torch.tensor(X, dtype=torch.float32)  # 再度テンソルに変換  
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]