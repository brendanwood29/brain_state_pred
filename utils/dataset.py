import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PyGDataset


class SingleSubjectBrainFuncRecursiveDataset(TorchDataset):
    def __init__(
        self,
        bold_data: np.ndarray,
        step: int,
        noise_strength: float = 0.1,
        tr: float | None = None,
        target_tr: float | None = None,
        scaler: Literal["min_max", "std"] | MinMaxScaler | StandardScaler | None = None,
    ):
        super().__init__()

        self.strength = noise_strength
        self.step = step
        data_length = bold_data.shape[0]

        formatted_scalers = {"min_max": MinMaxScaler, "std": StandardScaler}

        if target_tr is not None:
            assert tr is not None, "Tr must be specified to resample timeseries"
            new_len = int((data_length * tr) / target_tr)
            bold_data = signal.resample(bold_data, new_len, axis=0)  # type: ignore
            data_length = bold_data.shape[0]
        if scaler is None:
            pass
        elif isinstance(scaler, str):
            assert scaler in formatted_scalers, "scaler can only be `min_max` or `std`"
            scaler = formatted_scalers[scaler]
            bold_data = scaler.fit_transform(bold_data)  # type: ignore
            self._scaler = scaler
        else:
            bold_data = scaler.transform(bold_data)
        self.bold_data = torch.tensor(bold_data, dtype=torch.float)

        self.idxs = [
            (x, x + step) for x in range(step, data_length - step)
        ]  # Start at step so we can go back in time to the start of the sequence

    @property
    def data(self):
        return self.bold_data

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        start, end = self.idxs[idx]
        inp = self.bold_data[start:end].flatten()
        prev_inp = self.bold_data[[start - self.step]]
        out = self.bold_data[[end]]
        inp_noise = torch.normal(0, self.strength, inp.shape)
        return inp + inp_noise, out, prev_inp


class SingleSubjectBrainFuncDataset(TorchDataset):
    def __init__(
        self,
        bold_data: np.ndarray,
        step: int,
        noise_strength: float = 0.1,
        pred_len: int = 1,
        tr: float | None = None,
        target_tr: float | None = None,
        scaler: Literal["min_max", "std"] | MinMaxScaler | StandardScaler | None = None,
    ):
        super().__init__()

        self.inputs = []
        self.outputs = []
        self.strength = noise_strength
        data_length = bold_data.shape[0]

        formatted_scalers = {"min_max": MinMaxScaler, "std": StandardScaler}

        if target_tr is not None:
            assert tr is not None, "Tr must be specified to resample timeseries"
            new_len = int((data_length * tr) / target_tr)
            bold_data = signal.resample(bold_data, new_len, axis=0)  # type: ignore
            data_length = bold_data.shape[0]
        if scaler is None:
            pass
        elif isinstance(scaler, str):
            assert scaler in formatted_scalers, "scaler can only be `min_max` or `std`"
            scaler = formatted_scalers[scaler]
            bold_data = scaler.fit_transform(bold_data)  # type: ignore
            self._scaler = scaler
        else:
            bold_data = scaler.transform(bold_data)
        self.bold_data = bold_data
        for i in range(data_length):
            if (i + step + pred_len) <= data_length:
                self.inputs.append(
                    torch.tensor(bold_data[i : i + step], dtype=torch.float).flatten()
                )
                out = torch.tensor(bold_data[i + step], dtype=torch.float).unsqueeze(0)
                # out = torch.tensor(
                #     bold_data[i + step] - bold_data[i + step - 1], dtype=torch.float
                # ).flatten()
                # out = torch.tensor(
                #     bold_data[i + 1 : i + step + 1],
                #     dtype=torch.float,
                # ).flatten()
                # out = torch.tensor(
                #     bold_data[i + step : i + step + pred_len], dtype=torch.float
                # ).flatten()
                # out = torch.fft.rfft(out, dim=0)
                # out = torch.stack([out.real, out.imag], dim=0)
                self.outputs.append(out)
        # self.outputs = torch.stack(self.outputs, dim=0)
        # if mean is None and std is None:
        #     self.mean = self.outputs.mean(dim=0)
        #     self.std = self.outputs.std(dim=0)
        # else:
        #     self.mean = mean
        #     self.std = std
        # self.outputs = (self.outputs - self.mean) / torch.clamp(self.std, min=1e-9)

    @property
    def data(self):
        return self.bold_data

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        out = self.outputs[idx]
        inp_noise = torch.normal(0, self.strength, inp.shape)
        return inp + inp_noise, out


class SingleSubjectFFTFuncDataset(TorchDataset):
    def __init__(
        self,
        bold_data: np.ndarray,
        step: int,
        noise_strength: float = 0.1,
        pred_len: int = 1,
        tr: float | None = None,
        target_tr: float | None = None,
        mean: torch.Tensor | float = 0,
        std: torch.Tensor | float = 1,
    ):
        super().__init__()

        self.inputs = []
        self.outputs = []
        self.strength = noise_strength
        self.mean = mean
        self.std = std
        data_length = bold_data.shape[0]

        if target_tr is not None:
            assert tr is not None, "Tr must be specified to resample timeseries"
            new_len = int((data_length * tr) / target_tr)
            bold_data = signal.resample(bold_data, new_len, axis=0)  # type: ignore
            data_length = bold_data.shape[0]

        for i in range(data_length):
            if (i + step + pred_len) <= data_length:
                self.inputs.append(
                    torch.tensor(bold_data[i : i + step], dtype=torch.float).flatten()
                )
                out = torch.tensor(bold_data[i + 1 : i + step + 1], dtype=torch.float)
                # out = torch.tensor(
                #     bold_data[i + step : i + step + pred_len], dtype=torch.float
                # )
                out = torch.fft.rfft(out, dim=0)
                out = torch.stack([out.real, out.imag], dim=0)
                self.outputs.append(out)

        self.outputs = torch.stack(self.outputs, dim=0)  # B, Real/Imag, Coeffs, Region
        if not isinstance(self.mean, torch.Tensor) and not isinstance(
            self.std, torch.Tensor
        ):
            self.mean = self.outputs.mean(dim=0)
            self.std = self.outputs.std(dim=0)
            self.outputs = (self.outputs - self.mean) / (self.std + 1e-8)
        else:
            self.outputs = (self.outputs - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        out = self.outputs[idx]
        inp_noise = torch.normal(0, self.strength, inp.shape)
        return inp + inp_noise, out


class BrainFuncDataset(TorchDataset):
    def __init__(self, split_path: str | Path, step: int, strength: float):
        super().__init__()

        self.strength = strength
        self.inputs = []
        self.outputs = []
        with open(split_path, "r") as f:
            data = json.load(f)
        # TODO make this handle longitudinal data, or figure out a
        # good way to deal with it
        for subject in data:
            for ses in data[subject]:
                bold_data = pd.read_csv(
                    data[subject][ses]["file_path"], index_col=0
                ).to_numpy()
                data_length = bold_data.shape[0]
                for i in range(data_length):
                    if (i + step + 1) < data_length:
                        self.inputs.append(
                            torch.tensor(
                                bold_data[i : i + step], dtype=torch.float
                            ).flatten()
                        )
                        self.outputs.append(
                            torch.tensor(
                                bold_data[i + 1 : i + step + 1], dtype=torch.float
                            ).flatten()
                        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        out = self.outputs[idx]
        inp_noise = torch.normal(0, self.strength, inp.shape)
        return inp + inp_noise, out


class BrainFuncGCNDataset(PyGDataset):
    def __init__(
        self,
        split_path: str | Path,
        threshold: float,
        step: int,
    ):
        super().__init__()

        with open(split_path, "r") as f:
            data = json.load(f)

        self.dataset = []
        self.bold = []
        self.weights = []
        self.edge_idxs = []

        scan_num = 0
        for subject in data:
            for ses in data[subject]:
                bold_data = pd.read_csv(
                    data[subject][ses]["file_path"], index_col=0
                ).to_numpy()
                fc = pd.read_csv(
                    data[subject][ses]["file_path"].replace(
                        "cleaned-timeseries", "connectome"
                    ),
                    index_col=0,
                ).to_numpy()

                src, des = np.where(np.abs(fc) > threshold)
                edge_idx = np.stack([src, des])
                weights = fc[src, des]

                self.bold.append(torch.tensor(bold_data, dtype=torch.float))
                self.weights.append(torch.tensor(np.abs(weights), dtype=torch.float))
                self.edge_idxs.append(torch.tensor(edge_idx, dtype=torch.long))

                data_length = bold_data.shape[0]
                for i in range(data_length):
                    if (i + step) < data_length:
                        self.dataset.append((scan_num, i, i + step))
                scan_num += 1

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        ref = self.dataset[idx]

        return Data(
            x=self.bold[ref[0]][ref[1] : ref[2]].t(),
            edge_index=self.edge_idxs[ref[0]],
            edge_attr=self.weights[ref[0]].unsqueeze(-1),
            y=self.bold[ref[0]][ref[2]].unsqueeze(-1),
        )


class SingleSubjectBrainFuncGCNDataset(PyGDataset):
    def __init__(
        self,
        bold_data: np.ndarray,
        fc: np.ndarray,
        threshold: float,
        step: int,
    ):
        super().__init__()

        self.dataset = []
        src, des = np.where(np.abs(fc) > threshold)
        edge_idx = np.stack([src, des])
        weights = np.abs(fc[src, des])

        data_length = bold_data.shape[0]
        for i in range(data_length):
            if (i + step) < data_length:
                self.dataset.append(
                    Data(
                        x=torch.tensor(bold_data[i : i + step], dtype=torch.float).t(),
                        edge_index=torch.tensor(edge_idx, dtype=torch.long),
                        edge_attr=torch.tensor(weights, dtype=torch.float),
                        y=torch.tensor(bold_data[step], dtype=torch.float).unsqueeze(
                            -1
                        ),
                    )
                )

    def len(self):
        return len(self.dataset)

    def get(self, idx):

        return self.dataset[idx]


class SingleSubjectBrainFuncSTGCNDataset(TorchDataset):
    def __init__(
        self,
        bold_data: np.ndarray,
        fc: np.ndarray,
        threshold: float,
        step: int,
    ):
        super().__init__()

        self.dataset = []
        self.bold = torch.tensor(bold_data, dtype=torch.float)

        src, des = np.where(np.abs(fc) > threshold)
        edge_idx = np.stack([src, des])
        weights = fc[src, des]
        self.edge_idxs = torch.tensor(edge_idx, dtype=torch.long)
        self.weights = torch.tensor(np.abs(weights), dtype=torch.float)

        data_length = bold_data.shape[0]
        for i in range(data_length):
            if (i + step) < data_length:
                self.dataset.append((i, i + step))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ref = self.dataset[idx]

        return (
            self.bold[ref[0] : ref[1]].unsqueeze(-1),
            self.bold[ref[1]],
            self.edge_idxs,
            self.weights,
        )
