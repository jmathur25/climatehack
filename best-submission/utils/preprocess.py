from datetime import datetime, time, timedelta

import numpy as np
import xarray as xr
from numpy import float32
from tqdm import tqdm

SATELLITE_ZARR_PATH = "data/eumetsat_seviri_hrv_uk.zarr"


def main():
    dataset = xr.open_dataset(
        SATELLITE_ZARR_PATH,
        engine="zarr",
        chunks="auto",
    )

    times = dataset.get_index("time")
    min_date = times[0].date()
    max_date = times[-1].date()

    start_time = time(9, 0)
    end_time = time(16, 0)

    data = []
    date = min_date
    print(min_date, max_date, (max_date - min_date).days)
    with tqdm(total=(max_date - min_date).days) as pbar:

        # if you only want to preprocess a certain number of days, swap the while loop for the for loop below!
        # for _ in range(10):

        while date <= max_date:
            print(date)
            selection = (
                dataset["data"].sel(
                    time=slice(
                        datetime.combine(date, start_time),
                        datetime.combine(date, end_time),
                    ),
                )
                # comment out the .isel if you want the whole image
                .isel(
                    x=slice(550, 950),
                    y=slice(375, 700),
                )
            )

            if selection.shape == (85, 325, 400):
                data.append(selection.astype(float32).values)

            date += timedelta(days=1)
            pbar.update(1)

    clipped_data = np.clip(np.stack(data), 0.0, 1023.0)

    x_osgb = (
        dataset["x_osgb"]
        # comment out the .isel if you want the whole image
        .isel(
            x=slice(550, 950),
            y=slice(375, 700),
        ).values.astype(float32)
    )

    y_osgb = (
        dataset["y_osgb"]
        # comment out the .isel if you want the whole image
        .isel(
            x=slice(550, 950),
            y=slice(375, 700),
        ).values.astype(float32)
    )

    # you can also use np.savez_compressed if you'd rather compress everything!
    np.savez("data/sample", x_osgb=x_osgb, y_osgb=y_osgb, data=clipped_data)


if __name__ == "__main__":
    main()
