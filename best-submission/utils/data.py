from datetime import datetime, time, timedelta
from functools import lru_cache
from random import randrange
from typing import Iterator, T_co

import cv2
import numpy as np
import torch
import xarray as xr
from numpy import float32
from torch.utils.data import Dataset, IterableDataset

# Global Constants

# FORECAST = 24
INPUT_STEPS = 12
# BATCH_SIZE = 16

_MEAN_PIXEL = 240.3414
_STD_PIXEL = 146.52366

_MEAN_X_FLOW = -1.468137
_STD_X_FLOW = 2.535055

_MEAN_Y_FLOW = 0.200738
_STD_Y_FLOW = 1.133198

_MEAN_X_OSGB = 374844.9
_STD_X_OSGB = 134757.67

_MEAN_Y_OSGB = 438421.4
_STD_Y_OSGB = 215168.75

test_params = {
    "pyr_scale": 0.5,
    "levels": 2,
    "winsize": 40,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 0.7,
}


class ClimatehackDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        crops_per_slice: int = 1,
        with_osgb: bool = True,
    ) -> None:
        super().__init__()

        self.crops_per_slice = crops_per_slice
        self.with_osgb = with_osgb

        self._process_data(np.load(data_path))

    def _process_data(self, data):
        self.osgb_data = np.stack(
            [
                data["x_osgb"],
                data["y_osgb"],
            ]
        )

        self.cached_items = []
        data_array = data["data"]
        *_, t, y, x = data_array.shape
        for day in data_array:
            # change 4 (20 min) to whichever skip you like
            # this might depend on your memory constraints
            for i in range(0, t - 36, 4):
                input_slice = day[i : i + 12, :, :]
                target_slice = day[i + 12 : i + 36, :, :]

                crops = 0
                while crops < self.crops_per_slice:
                    crop = self._get_crop(input_slice, target_slice, y, x)
                    if crop is not None:
                        self.cached_items.append(crop)
                        crops += 1

    def _get_crop(self, input_slice, target_slice, y, x):
        rand_x = randrange(0, x - 128)
        rand_y = randrange(0, y - 128)

        osgb_data = self.osgb_data[:, rand_y : rand_y + 128, rand_x : rand_x + 128]
        input_data = input_slice[:, rand_y : rand_y + 128, rand_x : rand_x + 128]
        merged_data = (
            np.concatenate((osgb_data, input_data), 0) if self.with_osgb else input_data
        )
        target_data = target_slice[:, rand_y : rand_y + 128, rand_x : rand_x + 128]

        c = 14 if self.with_osgb else 12
        if merged_data.shape != (c, 128, 128) or target_data.shape != (24, 128, 128):
            return None

        return merged_data, target_data

    def __getitem__(self, index) -> T_co:
        return self.cached_items[index]

    def __len__(self) -> int:
        return len(self.cached_items)


class DaskDataset(IterableDataset):
    """
    This is a basic dataset class to help you get started with the Climate Hack.AI
    dataset. You are heavily encouraged to customise this to suit your needs.

    Notably, you will most likely want to modify this if you aim to train
    with the whole dataset, rather than just a small subset thereof.
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        start_date: datetime = None,
        end_date: datetime = None,
        crops_per_slice: int = 1,
        day_limit: int = 0,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.crops_per_slice = crops_per_slice
        self.day_limit = day_limit
        self.cached_items = []

        times = self.dataset.get_index("time")
        self.min_date = times[0].date()
        self.max_date = times[-1].date()

        if start_date is not None:
            self.min_date = max(self.min_date, start_date)

        if end_date is not None:
            self.max_date = min(self.max_date, end_date)
        elif self.day_limit > 0:
            self.max_date = min(
                self.max_date, self.min_date + timedelta(days=self.day_limit)
            )

    def _image_times(self, start_time, end_time):
        date = self.min_date
        while date <= self.max_date:
            current_time = datetime.combine(date, start_time)
            while current_time.time() <= end_time:
                yield current_time
                current_time += timedelta(minutes=20)

            date += timedelta(days=1)

    def _get_crop(self, input_slice, target_slice):
        # roughly over the mainland UK
        rand_x = randrange(550, 950 - 128)
        rand_y = randrange(375, 700 - 128)

        # make a data selection
        selection = input_slice.isel(
            x=slice(rand_x, rand_x + 128),
            y=slice(rand_y, rand_y + 128),
        )

        # get the OSGB coordinate data
        osgb_data = np.stack(
            [
                selection["x_osgb"].values.astype(float32),
                selection["y_osgb"].values.astype(float32),
            ]
        )

        if osgb_data.shape != (2, 128, 128):
            return None

        # get the input satellite imagery
        input_data = selection["data"].values.astype(float32)
        if input_data.shape != (12, 128, 128):
            return None

        # get the target output
        target_output = (
            target_slice["data"]
            .isel(
                x=slice(rand_x + 32, rand_x + 96),
                y=slice(rand_y + 32, rand_y + 96),
            )
            .values.astype(float32)
        )

        if target_output.shape != (24, 64, 64):
            return None

        return osgb_data, input_data, target_output

    def __iter__(self) -> Iterator[T_co]:
        if self.cached_items:
            for item in self.cached_items:
                yield item

            return

        start_time = time(9, 0)
        end_time = time(14, 0)

        for current_time in self._image_times(start_time, end_time):
            data_slice = self.dataset.loc[
                {
                    "time": slice(
                        current_time,
                        current_time + timedelta(hours=2, minutes=55),
                    )
                }
            ]

            if data_slice.sizes["time"] != 36:
                continue

            input_slice = data_slice.isel(time=slice(0, 12))
            target_slice = data_slice.isel(time=slice(12, 36))

            crops = 0
            while crops < self.crops_per_slice:
                crop = self._get_crop(input_slice, target_slice)
                if crop:
                    self.cached_items.append(crop)
                    yield crop

                crops += 1


class CustomDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, path, with_osgb=False, random_state=7):

        if with_osgb:
            raise ValueError(
                "OSGB data not supported. Please use ClimatehackDataset instead."
            )

        data_array = np.load(path)
        self.data = data_array["data"]
        self.times = data_array["times"]

        self.generator = np.random.RandomState(random_state)

    def _check_times(self, tstart, tend):
        return int((tend - tstart) / np.timedelta64(1, "m")) == 175

    def _get_crop(self, data):
        # roughly over the mainland UK
        rand_x = self.generator.randint(0, data.shape[2] - 128)
        rand_y = self.generator.randint(0, data.shape[1] - 128)
        # make a data selection
        return data[:, rand_y : rand_y + 128, rand_x : rand_x + 128]

    def __getitem__(self, index):
        tend = self.times[index + 35]
        tstart = self.times[index]
        if not self._check_times(tstart, tend):
            return self.__getitem__((index + 35) % len(self))
        all_data = self.data[index : index + 12 + 24]
        all_data = self._get_crop(all_data)

        #         -1 means rotate cw 90, 0 means no rotation, 1 means ccw 90, 2 means ccw 180
        #         rot_amount = self.generator.randint(-1, 3)
        #         if rot_amount != 0:
        #             all_data = np.rot90(all_data, k=rot_amount, axes=(1, 2)).copy()
        #         generate contrast and brightness changes
        #         cf = np.random.uniform(low=0.7, high=1.3)
        #         bf = np.random.uniform(low=-50, high=50)
        #         all_data = all_data * cf + bf

        all_data = torch.FloatTensor(all_data)
        x = all_data[:INPUT_STEPS]
        y = all_data[INPUT_STEPS:]
        return x, y

    def __len__(self):
        return len(self.times) - 35


class OSGBDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, path, coords_path="data/coords.npz", random_state=7):

        data_array = np.load(path)
        coords = np.load(coords_path)

        self.times = data_array["times"]
        self.data = data_array["data"]
        self.x_osgb = coords["x_osgb"]
        self.y_osgb = coords["y_osgb"]
        self.generator = np.random.RandomState(random_state)

    def _check_times(self, tstart, tend):
        return int((tend - tstart) / np.timedelta64(1, "m")) == 175

    def _get_crop(self, data):
        # roughly over the mainland UK
        rand_x = self.generator.randint(0, data.shape[2] - 128)
        rand_y = self.generator.randint(0, data.shape[1] - 128)
        # make a data selection
        return (
            data[:, rand_y : rand_y + 128, rand_x : rand_x + 128],
            self.x_osgb[rand_y : rand_y + 128, rand_x : rand_x + 128],
            self.y_osgb[rand_y : rand_y + 128, rand_x : rand_x + 128],
        )

    def __getitem__(self, index):
        tend = self.times[index + 35]
        tstart = self.times[index]
        if not self._check_times(tstart, tend):
            return self.__getitem__((index + 35) % len(self))
        all_data = self.data[index : index + INPUT_STEPS + 24]
        all_data, osgb_x, osgb_y = self._get_crop(all_data)
        flow = cv2.calcOpticalFlowFarneback(
            prev=all_data[INPUT_STEPS - 1],
            next=all_data[INPUT_STEPS - 2],
            flow=None,
            **test_params,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        flow = -flow

        osgb_x = (osgb_x - _MEAN_X_OSGB) / _STD_X_OSGB
        osgb_y = (osgb_y - _MEAN_Y_OSGB) / _STD_Y_OSGB
        all_data = torch.FloatTensor(all_data)
        osgb_x = torch.FloatTensor(osgb_x)
        osgb_y = torch.FloatTensor(osgb_y)
        x = (all_data[:INPUT_STEPS] - _MEAN_PIXEL) / _STD_PIXEL
        y = all_data[INPUT_STEPS:]

        flow_x = (flow[:, :, 0] - _MEAN_X_FLOW) / _STD_X_FLOW
        flow_y = (flow[:, :, 1] - _MEAN_Y_FLOW) / _STD_Y_FLOW

        flow_x = torch.FloatTensor(flow_x)
        flow_y = torch.FloatTensor(flow_y)

        x = torch.cat(
            [
                x,
                osgb_x.unsqueeze(0),
                osgb_y.unsqueeze(0),
                flow_x.unsqueeze(0),
                flow_y.unsqueeze(0),
            ],
            dim=0,
        )
        return x, y

    def __len__(self):
        return len(self.times) - 35
