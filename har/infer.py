import json
import numpy as np
import os
import time
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from har.model import Model, ModelArgs
device = torch.device('cpu')

label_list = ['climbing up', 'climbing down', 'walking', 'running', 'static', 'unknown']
mapping = {
    0: 'climbing up',
    1: 'climbing down',
    2: 'walking',
    3: 'running',
    4: 'static',
    5: 'unknown'
}

class IMUDataset(Dataset):
    def __init__(self, data):
        # data 为 numpy 数组，转换为 tensor
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx, :6, :] # only gyro and acc

def infer(model, df: pd.DataFrame):
    try:
        window_size = 120
        input_data = []
        start_time = time.time()
        for i in range(0, len(df) - window_size + 1, window_size):
            window = df.iloc[i:i+window_size]
            input_data.append(window[['gyro_X', 'gyro_Y', 'gyro_Z', 'acc_X', 'acc_Y', 'acc_Z', 'mag_X', 'mag_Y', 'mag_Z']].values)

        df.drop(['gyro_X', 'gyro_Y', 'gyro_Z', 'acc_X', 'acc_Y', 'acc_Z', 'mag_X', 'mag_Y', 'mag_Z'], axis=1, inplace=True)
        # save rest data index
        rest_idxes = df.iloc[len(df) - len(df) % window_size:].index
        input_data = np.array(input_data, dtype=float) # [bs, 120, 9]

        predictions = []
        
        # Create a DataLoader for batch processing
        batch_size = 512  # You can adjust this based on your memory constraints
        dataset = IMUDataset(input_data) # returns a tensor
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        unknown_tensor = torch.tensor(5, device=device)

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                imu_inputs = batch
                imu_inputs = imu_inputs.to(device, non_blocking=True)

                # Process the entire batch at once
                outputs = model(imu_inputs)

                logits = torch.softmax(outputs, dim=-1)
                threshold = 0.5
                unknown_mask = torch.max(logits, dim=-1).values < threshold
                pred_indices = torch.where(
                    unknown_mask,
                    unknown_tensor,
                    outputs.argmax(dim=-1)
                )
                predictions.extend(pred_indices.cpu().numpy().tolist())
        
        pred_labels = [mapping.get(idx, 'unknown') for idx in predictions]
        activity_col = []
        for label in pred_labels:
            activity_col.extend([label] * window_size)
        
        # 对于剩余不足一个窗口的部分，统一标记为 'unknown'
        if remainder_idx.size > 0:
            activity_col.extend(['unknown'] * len(remainder_idx))
        
        # 检查长度是否匹配原始数据行数
        if len(activity_col) != len(df) + len(remainder_idx):
            # 注意：df 已经删除了 sensor 列，但行数不变
            total_expected = num_full_windows * window_size + len(remainder_idx)
            if len(activity_col) != total_expected:
                raise ValueError("activity 列长度不匹配原始数据行数")
        # 添加 activity 列到 df
        df['activity'] = activity_col # ["timestamp", "activity"]
        total_time = time.time() - start_time
        return df, total_time
    except Exception as e:
        raise e