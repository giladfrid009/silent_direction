from src.utils.logging import create_logger
from src.data import TableLoader
from typing import Any
import datasets
from datasets import DatasetDict, Dataset
from enum import Enum
import pandas as pd


logger = create_logger(__name__)


class DatasetName(str, Enum):
    HH_RLHF = "hh-rlhf"
    SLIM_ORCA = "slim-orca"


SUPPORTED_DATASETS = [e.value for e in DatasetName]


# TODO: SHOULD ALSO RETURN A PROPERTY WHETHER IS CHAT DATASET OR RAW STRING DATASET
def load_raw_dataset(name: str) -> DatasetDict:
    """
    Loads and returns a dataset by name. Each dataset is expected to have the following columns:
    - "prompt": The input user prompt to the model.
        Either a string or a list of alternating user/assistant messages.

    If the dataset does not have predefined splits, it will be loaded as a single "train" split.
    The user can then split it into training, validation, and test sets as needed.

    Args:
        name (str): The name of the dataset to load. Must be one of the supported datasets.

    Returns:
        DatasetDict: A dictionary-like object containing the dataset splits.
    """
    # validate alternating roles structure
    def filter_fn(example: Any) -> bool:
        for col in ["chosen", "rejected"]:
            conv = example[col]
            for i, msg in enumerate(conv):
                if i % 2 == 0 and msg["role"] != "user":
                    return False
                if i % 2 == 1 and msg["role"] != "assistant":
                    return False
        return True

    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {name}. Supported datasets are: {SUPPORTED_DATASETS}")

    if name == DatasetName.HH_RLHF:
        ds_train: Dataset = datasets.load_dataset("trl-internal-testing/hh-rlhf-trl-style", split="train")  # type: ignore

        ds_train = ds_train.filter(filter_fn)
        ds_train = ds_train.rename_columns({"chosen": "prompt", "prompt": "input"})

        # # split eval from train, eval of size 2000
        # split_dict = ds_train.train_test_split(test_size=2000, seed=42)
        # ds_rest = split_dict["train"]
        # ds_eval = split_dict["test"]

        # split_dict = ds_rest.train_test_split(train_size=10000, test_size=2000, seed=43)
        # ds_train = split_dict["train"]
        # ds_test = split_dict["test"]

        # return DatasetDict({"train": ds_train, "validation": ds_eval, "test": ds_test})

    elif name == DatasetName.SLIM_ORCA:
        ds_train: Dataset = datasets.load_dataset("Open-Orca/SlimOrca", split="train")  # type: ignore

        role_map = {"human": "user", "gpt": "assistant"}

        def convert_fn(example: Any) -> dict:
            # Filter out system messages and convert to role/content format
            conv = [
                {"role": role_map[msg["from"]], "content": msg["value"]}
                for msg in example["conversations"]
                if msg["from"] in role_map
            ]
            return {"prompt": conv}

        ds_train = ds_train.map(convert_fn, remove_columns=["conversations"])
        ds_train = ds_train.filter(filter_fn)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # split eval from train, eval of size 2000
    split_dict = ds_train.train_test_split(test_size=2000, seed=42)
    ds_rest = split_dict["train"]
    ds_eval = split_dict["test"]

    split_dict = ds_rest.train_test_split(train_size=10000, test_size=2000, seed=43)
    ds_train = split_dict["train"]
    ds_test = split_dict["test"]

    return DatasetDict({"train": ds_train, "validation": ds_eval, "test": ds_test})


def load_dataset(name: str, shuffle: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads a dataset by name and returns the training, validation, and test sets as pandas DataFrames.

    Args:
        names (list[str]): List of dataset names to load. Each name must be one of the supported datasets.
        shuffle (bool): Whether to shuffle the dataset splits. Default is True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and test datasets as pandas DataFrames.
    """

    ds_dict = load_raw_dataset(name)

    ds_train = ds_dict["train"]
    ds_val = ds_dict["validation"] if "validation" in ds_dict else None
    ds_test = ds_dict["test"] if "test" in ds_dict else None

    if ds_val is None and ds_test is None:
        ds_val = ds_train.shuffle(seed=0, keep_in_memory=True)
        ds_test = ds_train.shuffle(seed=1, keep_in_memory=True)
        logger.info(f"Dataset {name} has no validation or test set, using training set for both.")

    if ds_val is None and ds_test is not None:
        ds_val = ds_test.shuffle(seed=2, keep_in_memory=True)
        logger.info(f"Dataset {name} has no validation set, using test set as validation set.")

    if ds_test is None and ds_val is not None:
        ds_test = ds_val.shuffle(seed=3, keep_in_memory=True)
        logger.info(f"Dataset {name} has no test set, using validation set as test set.")

    if ds_val is None or ds_test is None:
        raise ValueError(f"Dataset {name} is missing validation or test set and could not be created from training set.")

    if shuffle:
        ds_train = ds_train.shuffle(keep_in_memory=True)  # type: ignore
        ds_val = ds_val.shuffle(keep_in_memory=True)  # type: ignore
        ds_test = ds_test.shuffle(keep_in_memory=True)  # type: ignore

    return (
        pd.DataFrame(ds_train.to_dict()),
        pd.DataFrame(ds_val.to_dict()),
        pd.DataFrame(ds_test.to_dict()),
    )


def is_chat_dataset(data: TableLoader | pd.DataFrame) -> bool:
    """
    Determines whether the given dataset is a chat dataset, or a raw string dataset.

    Args:
        data (TableLoader | pd.DataFrame): The dataset to check. Can be either a TableLoader or a pandas DataFrame.

    Returns:
        bool: True if the dataset is a chat dataset
            (i.e. "prompt" field is a list of alternating user/assistant messages), False if the dataset is a raw string dataset (i.e. "prompt" field is a string).
    """

    if isinstance(data, pd.DataFrame):
        data = TableLoader(data, batch_size=2, shuffle=False)

    sample = next(iter(data))["prompt"][0]

    if isinstance(sample, str):
        return False

    if isinstance(sample, list) and all(isinstance(item, dict) for item in sample):
        return True

    raise ValueError("Unexpected format for 'prompt' field. Expected either a string or a list of dictionaries.")
