import random
from typing import Any, Generator
import pandas as pd


class TableLoader:
    """
    Iterate over a DataFrame in row-wise batches.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Precompute index list and number of samples
        self._indices = list(df.index)
        self._n_samples = len(self._indices)
        self._n_batches = self._compute_num_batches()

    def get_hparams(self) -> dict:
        """
        Returns the hyperparameters of the TableLoader as a dictionary.
        """
        return {
            "num_samples": self._n_samples,
            "num_batches": self._n_batches,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }

    def validate(self, columns: list[str]) -> None:
        """
        Public method: check that each column in `columns` exists in `self.df.columns`.

        Raises:
            ValueError: if any requested column is not found in self.df.
        """
        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

    @property
    def n_samples(self) -> int:
        """
        Number of samples in the DataFrame.
        """
        return self._n_samples

    def _compute_num_batches(self) -> int:
        if self.drop_last:
            return self._n_samples // self.batch_size
        else:
            # include a final smaller batch if necessary
            return (self._n_samples + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        """
        Number of batches (independent of which columns you slice out).
        """
        return self._n_batches

    def __iter__(self) -> Generator[dict[str, list[Any]], None, None]:
        columns = self.df.columns.tolist()

        # Make a copy of the indices and shuffle if requested
        idxs = self._indices.copy()
        if self.shuffle:
            random.shuffle(idxs)

        # Now slice out row-indices batch by batch
        for batch_idx in range(self._n_batches):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_idxs = idxs[start:end]

            # If drop_last=True and the final batch is too small, stop early
            if len(batch_idxs) < self.batch_size and self.drop_last:
                break

            data = {col: self.df.loc[batch_idxs, col].tolist() for col in columns}
            yield data

    def copy(self, **kwargs) -> "TableLoader":
        """
        Create and return a shallow-copy of this TableLoader. The new instance
        wraps the same DataFrame (no deep copy of df), but has its own attributes
        (columns, batch_size, shuffle, drop_last).

        Args:
            **kwargs: Additional keyword arguments to override attributes in the new instance.
        """
        attrs = {
            "df": self.df,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }

        for key, value in kwargs.items():
            if key in attrs:
                attrs[key] = value
            else:
                raise ValueError(f"Invalid attribute '{key}' for TableLoader")

        return TableLoader(**attrs)


class TableIterator:
    """
    A simple wrapper around TableLoader to provide an iterator interface.
    """

    def __init__(
        self,
        table_loader: TableLoader,
        num_batches: int | None = None,
        num_epochs: int | None = 1,
    ) -> None:

        total_steps = len(table_loader)
        if num_epochs is not None and num_batches is not None:
            total_steps = min(num_batches, num_epochs * len(table_loader))
        elif num_epochs is not None:
            total_steps = num_epochs * len(table_loader)
        elif num_batches is not None:
            total_steps = num_batches

        self.table_loader = table_loader
        self.total_steps = total_steps
        self.current_step = 0

    def __iter__(self) -> Generator[dict[str, list[Any]], None, None]:
        while self.current_step < self.total_steps:
            for batch in self.table_loader:
                if self.current_step >= self.total_steps:
                    break
                yield batch
                self.current_step += 1

    def __len__(self) -> int:
        return self.total_steps
