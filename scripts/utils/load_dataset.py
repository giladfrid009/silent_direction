import os
from typing import Any
from enum import Enum
import pandas as pd
from functools import partial
from tqdm.auto import tqdm
import datasets
from datasets import DatasetDict, Dataset, IterableDataset
from src.utils.logging import create_logger
from src.data import TableLoader


logger = create_logger(__name__)


class DatasetName(str, Enum):
    HH_RLHF = "hh-rlhf"
    SLIM_ORCA = "slim-orca"
    TULU_V2 = "tulu-v2"
    OASST2 = "oasst2"
    LMSYS_1M = "lmsys-1m"


SUPPORTED_DATASETS = [e.value for e in DatasetName]


def dataset_dir(name: str) -> str:
    """
    returns `<this-file-path>/../../../data/{name}`
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "..", "data", name)


def materialize_dataset(it_ds: IterableDataset, total: int | None = None) -> Dataset:
    items = []
    for item in tqdm(it_ds, desc="Materializing Dataset", total=total):
        items.append(item)
    return Dataset.from_list(items)


# validate alternating roles structure
def validate_conversation(example: Any, allow_system: bool = False) -> bool:
    conv = example["prompt"]
    if len(conv) == 0:
        return False

    if allow_system and conv[0]["role"] == "system":
        conv = conv[1:]  # skip system message

    for i, msg in enumerate(conv):
        if i % 2 == 0 and msg["role"] != "user":
            return False
        if i % 2 == 1 and msg["role"] != "assistant":
            return False
    return True


def split_dataset(ds: Dataset, train_size: int, eval_size: int, test_size: int, seed: int = 42) -> DatasetDict:
    split_dict = ds.train_test_split(test_size=eval_size, seed=seed)
    ds_rest = split_dict["train"]
    ds_eval = split_dict["test"]

    split_dict = ds_rest.train_test_split(train_size=train_size, test_size=test_size, seed=seed + 1)
    ds_train = split_dict["train"]
    ds_test = split_dict["test"]

    return DatasetDict({"train": ds_train, "validation": ds_eval, "test": ds_test})


def load_hh_rlhf() -> DatasetDict:
    data_directory = dataset_dir("hh-rlhf")
    if os.path.exists(data_directory):
        logger.info(f"Loading dataset from disk at {data_directory}")
        return DatasetDict.load_from_disk(data_directory)

    ds_train = datasets.load_dataset("trl-internal-testing/hh-rlhf-trl-style", split="train", streaming=True)  # type: ignore
    ds_train = ds_train.remove_columns("prompt").rename_column("chosen", "prompt")
    ds_train = ds_train.filter(partial(validate_conversation, allow_system=False))
    ds_train = materialize_dataset(ds_train, total=161_000)

    ds_dict = split_dataset(ds_train, train_size=50_000, eval_size=5000, test_size=5000)
    ds_dict.save_to_disk(data_directory)
    logger.info(f"Saving dataset to disk at {data_directory}")
    return ds_dict


def load_slim_orca() -> DatasetDict:
    data_directory = dataset_dir("slim-orca")
    if os.path.exists(data_directory):
        logger.info(f"Loading dataset from disk at {data_directory}")
        return DatasetDict.load_from_disk(data_directory)

    # Filter out system messages and convert to role/content format
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}

    def map_fn(example: Any) -> dict:
        conv = [
            {"role": role_map[msg["from"]], "content": msg["value"]}
            for msg in example["conversations"]
            if (msg["from"] != "system" or (msg["from"] == "system" and "assistant" not in msg["value"]))
        ]
        return {"prompt": conv}

    ds_train = datasets.load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)  # type: ignore
    ds_train = ds_train.map(map_fn, remove_columns=["conversations"])
    ds_train = ds_train.filter(partial(validate_conversation, allow_system=True))
    ds_train = materialize_dataset(ds_train, total=518_000)

    ds_dict = split_dataset(ds_train, train_size=50_000, eval_size=5000, test_size=5000)
    logger.info(f"Saving dataset to disk at {data_directory}")
    ds_dict.save_to_disk(data_directory)
    return ds_dict


def load_tulu_v2() -> DatasetDict:
    data_directory = dataset_dir("tulu-v2")
    if os.path.exists(data_directory):
        logger.info(f"Loading dataset from disk at {data_directory}")
        return DatasetDict.load_from_disk(data_directory)

    def map_fn(example: Any) -> dict:
        conv = [{"role": msg["role"], "content": msg["content"]} for msg in example["messages"] if msg["role"] in ["user", "assistant"]]
        return {"prompt": conv}

    ds_train = datasets.load_dataset("allenai/tulu-v2-sft-mixture", split="train", streaming=True)  # type: ignore
    ds_train = ds_train.map(map_fn, remove_columns=["dataset", "id", "messages"])
    ds_train = ds_train.filter(partial(validate_conversation, allow_system=False))
    ds_train = materialize_dataset(ds_train, total=326_000)

    ds_dict = split_dataset(ds_train, train_size=50_000, eval_size=5000, test_size=5000)
    logger.info(f"Saving dataset to disk at {data_directory}")
    ds_dict.save_to_disk(data_directory)
    return ds_dict


def load_lmsys_1m() -> DatasetDict:
    data_directory = dataset_dir("lmsys-1m")
    if os.path.exists(data_directory):
        logger.info(f"Loading dataset from disk at {data_directory}")
        return DatasetDict.load_from_disk(data_directory)

    ds_train = datasets.load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)  # type: ignore
    ds_train = ds_train.rename_column("conversation", "prompt")
    ds_train = ds_train.filter(partial(validate_conversation, allow_system=False))
    ds_train = materialize_dataset(ds_train, total=1_000_000)

    ds_dict = split_dataset(ds_train, train_size=50_000, eval_size=5000, test_size=5000)
    logger.info(f"Saving dataset to disk at {data_directory}")
    ds_dict.save_to_disk(data_directory)
    return ds_dict


def load_oasst2() -> DatasetDict:
    data_directory = dataset_dir("oasst2")
    if os.path.exists(data_directory):
        logger.info(f"Loading dataset from disk at {data_directory}")
        return DatasetDict.load_from_disk(data_directory)

    all_msgs = datasets.load_dataset("OpenAssistant/oasst2", split="train", streaming=True)

    # Keep only ready-for-export, non-deleted messages
    all_msgs = all_msgs.filter(lambda ex: ex["tree_state"] == "ready_for_export" and not ex["deleted"])
    df = pd.DataFrame(materialize_dataset(all_msgs, total=129_000).to_dict())

    # Identify English trees by their root message language
    root_df = df[df["message_id"] == df["message_tree_id"]]
    tree_ids = set(root_df["message_tree_id"].tolist())

    # Build message lookup and parent -> children map
    msg_lookup = df.set_index("message_id").to_dict("index")
    children_map: dict[str, list[str]] = {}
    for msg_id, parent_id in zip(df["message_id"], df["parent_id"]):
        if not isinstance(parent_id, str):
            continue
        if parent_id not in children_map:
            children_map[parent_id] = []
        children_map[parent_id].append(msg_id)

    # Sort children by rank (lower rank = higher quality; unranked last)
    def get_rank(mid: str) -> float:
        rank = msg_lookup[mid]["rank"]
        if rank is None or (isinstance(rank, float) and pd.isna(rank)):
            return float("inf")
        return float(rank)

    for parent in children_map:
        children_map[parent].sort(key=get_rank)

    # DFS to extract all linear conversation paths (root → leaf)
    def extract_paths(msg_id: str, path: list[str]) -> list[list[str]]:
        path = path + [msg_id]
        children = children_map.get(msg_id, [])
        if not children:
            return [path]
        result = []
        for child_id in children:
            result.extend(extract_paths(child_id, path))
        return result

    conversations = []
    for tree_id in tree_ids:
        if tree_id not in msg_lookup:
            continue
        for path in extract_paths(tree_id, []):
            conv = [
                {
                    "role": "user" if msg_lookup[mid]["role"] == "prompter" else "assistant",
                    "content": msg_lookup[mid]["text"],
                }
                for mid in path
            ]
            conversations.append({"prompt": conv})

    ds_train = Dataset.from_list(conversations)
    ds_train = ds_train.filter(partial(validate_conversation, allow_system=False))

    ds_dict = split_dataset(ds_train, train_size=15_000, eval_size=2500, test_size=2500)
    logger.info(f"Saving dataset to disk at {data_directory}")
    ds_dict.save_to_disk(data_directory)
    return ds_dict


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

    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {name}. Supported datasets are: {SUPPORTED_DATASETS}")

    if name == DatasetName.HH_RLHF:
        return load_hh_rlhf()

    elif name == DatasetName.SLIM_ORCA:
        return load_slim_orca()

    elif name == DatasetName.TULU_V2:
        return load_tulu_v2()

    elif name == DatasetName.OASST2:
        return load_oasst2()

    elif name == DatasetName.LMSYS_1M:
        return load_lmsys_1m()

    else:
        raise ValueError(f"Unsupported dataset: {name}")


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
