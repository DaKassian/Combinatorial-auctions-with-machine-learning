from torch.utils.data import Dataset
import torch.utils.data
import numpy as np


class DataSet(Dataset):
    def __init__(self, data, num_bids):
        # self.data = self.normalize_data(data)
        self.data = data
        self.num_bids = num_bids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # solution_columns = [col for col in self.data.columns if col.startswith('Solution_')]
        # input_columns = [col for col in self.data.columns if not col.startswith('Solution_')]
        # return torch.tensor(self.data.loc[idx].loc[input_columns]), torch.tensor(
        #     self.data.loc[idx].loc[solution_columns])
        #
        return torch.tensor(self.data.iloc[idx].iloc[:-self.num_bids]), torch.tensor(
            self.data.iloc[idx].iloc[-self.num_bids:])

    def normalize_data(self, data):
        # Apply Min-Max normalization to each feature
        for column in data.columns:
            if (data[column].max() - data[column].min()) != 0:
                data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
        return data
