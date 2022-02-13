import datetime
from functools import lru_cache
from typing import Iterator, T_co

import numpy as np
import xarray as xr
from torch.utils.data import Dataset


class ClimatehackDataset(Dataset):
    """ """

    def __init__(
        self,
        dataset: xr.Dataset,
        random_state: int,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.generator = np.random.RandomState(random_state)

        times = self.dataset.get_index("time")
        self.min_date = times[0].date()
        self.max_date = times[-1].date()

        if start_date is not None:
            self.min_date = max(self.min_date, start_date)

        if end_date is not None:
            self.max_date = min(self.max_date, end_date)

        self.start_hour = 9
        self.end_hour = 14  # start of a window is 2pm, forecasts to 5pm

        total_days = (self.max_date - self.min_date).days
        self.days = np.arange(total_days)

    def _random_image_times(self, start_hour, end_hour):
        total_days = (self.max_date - self.min_date).days
        shuffled_days = np.arange(total_days)
        self.generator.shuffle(shuffled_days)

        # make 2 choices per day
        middle = (end_hour - start_hour) // 2 + start_hour
        for day in shuffled_days:
            date = self.min_date + datetime.timedelta(days=int(day))
            hour1 = self.generator.randint(start_hour, middle + 1)
            hour2 = self.generator.randint(middle + 1, end_hour + 1)
            hour1 = datetime.time(hour1)
            hour2 = datetime.time(hour2)

            current_time = datetime.datetime.combine(date, hour1)
            yield current_time

            current_time = datetime.datetime.combine(date, hour2)
            yield current_time

    def _get_crop(self, input_slice, target_slice):
        # roughly over the mainland UK
        rand_x = self.generator.randint(550, 950 - 128)
        rand_y = self.generator.randint(375, 700 - 128)

        # make a data selection
        in_crop = input_slice[:, rand_y : rand_y + 128, rand_x : rand_x + 128]
        target_crop = target_slice[
            :, rand_y + 32 : rand_y + 96, rand_x + 32 : rand_x + 96
        ]

        return in_crop, target_crop

    def __getitem__(self, i):
        day = self.days[i]
        time = datetime.time(self.generator.randint(self.start_hour, self.end_hour + 1))
        date = self.min_date + datetime.timedelta(days=int(day))
        date_and_time = datetime.datetime.combine(date, time)
        # get a 3h slice
        data_slice = self.dataset.loc[
            {
                "time": slice(
                    date_and_time,
                    date_and_time + datetime.timedelta(hours=2, minutes=55),
                )
            }
        ]

        # need 36 items
        if data_slice.sizes["time"] != 36:
            # otherwise just do the next day
            return self.__getitem__(i + 1)

        # get the full images
        data = data_slice["data"].values
        input_data = data[:12]
        target_data = data[12:]

        return self._get_crop(input_data, target_data)

    # def __iter__(self) -> Iterator[T_co]:
    #     start_hour = 9
    #     end_hour = 14  # start of a window is 2pm, forecasts to 5pm

    #     for current_time in self._random_image_times(start_hour, end_hour):
    #         data_slice = self.dataset.loc[
    #             {
    #                 "time": slice(
    #                     current_time,
    #                     current_time + timedelta(hours=2, minutes=55),
    #                 )
    #             }
    #         ]

    #         if data_slice.sizes["time"] != 36:
    #             continue

    #         # download the full images
    #         data = data_slice["data"].values
    #         input_data = data[:12]
    #         target_data = data[12:]

    #         crops = 0
    #         while crops < self.crops_per_slice:
    #             crop = self._get_crop(input_data, target_data)
    #             if crop:
    #                 yield crop

    #             crops += 1

    def __len__(self):
        return len(self.days)
