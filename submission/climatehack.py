import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


class BaseEvaluator:
    def __init__(self) -> None:
        self.setup()

    def setup(self):
        """Sets up anything required for evaluation, e.g. models."""
        pass

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """

        raise NotImplementedError(
            "You need to extend this class to use your trained model(s)."
        )

    def _get_io_paths(self) -> Tuple[Path, Path]:
        """Gets the input and output directory paths from DOXA.

        Returns:
            Tuple[Path, Path]: The input and output paths
        """
        try:
            return Path(sys.argv[1]), Path(sys.argv[2])
        except IndexError:
            raise Exception(
                f"Run using: {sys.argv[0]} [input directory] [output directory]"
            )

    def _get_group_path(self) -> str:
        """Gets the path for the next group to be processed.

        Raises:
            ValueError: An unknown message was received from DOXA.

        Returns:
            str: The path of the next group.
        """

        msg = input()
        if not msg.startswith("Process "):
            raise ValueError(f"Unknown messsage {msg}")

        return msg[8:]

    def _evaluate_group(self, group: dict) -> List[np.ndarray]:
        """Evaluates a group of satellite image sequences using
        the user-implemented model(s).

        Args:
            group (dict): The OSGB and satellite imagery data.

        Returns:
            List[np.ndarray]: The predictions.
        """

        return [self.predict(*datum) for datum in zip(group["osgb"], group["data"])]

    def evaluate(self):
        """Evaluates the user's model on DOXA.

        Messages are sent and received through stdio.

        The input data is loaded from a directory in groups.

        The predictions are written to another directory in groups.

        Raises:
            Exception: An error occurred somewhere.
        """

        print("STARTUP")
        input_path, output_path = self._get_io_paths()

        # process test data groups
        while True:
            # load the data for the group DOXA requests
            group_path = self._get_group_path()
            group_data = np.load(input_path / group_path)

            # make predictions for this group
            try:
                predictions = self._evaluate_group(group_data)
            except Exception as err:
                raise Exception(f"Error while processing {group_path}: {str(err)}")

            # save the output group predictions
            np.savez(
                output_path / group_path,
                data=np.stack(predictions),
            )
            print(f"Exported {group_path}")
