import bdb
from dataclasses import dataclass
import functools
import logging
import time

import numpy as np
import torch

from constants import DATASET_ERROR_VERBOSITY


@dataclass
class DataloadFailure:
    index: int
    msg: str
    err_type: str


def retry_random_idx_on_err(verbosity=DATASET_ERROR_VERBOSITY, do_retry=True):

    assert 0 <= verbosity <= 3

    rng = None
    subprocess_seed_set = False

    def decorator(getitem_fn):
        MAX_RETRIES = 100
        BACKOFF_THRESHOLD = 10

        @functools.wraps(getitem_fn)
        def getitem_wrapper(self, index):

            nonlocal rng, do_retry, subprocess_seed_set
            do_retry &= not getattr(self, 'is_eval', False)  # Try to avoid retrying on eval splits
            in_subprocess = torch.utils.data.get_worker_info() is not None

            if rng is None or (in_subprocess and not subprocess_seed_set):
                rng = np.random.RandomState()
                rng.set_state(np.random.get_state())
                subprocess_seed_set = in_subprocess

            for retry in range(MAX_RETRIES):
                if retry > BACKOFF_THRESHOLD:
                    backoff_duration = (retry - BACKOFF_THRESHOLD) ** 2
                    logging.info(
                        f"Dataload retry {retry}: backing off for {backoff_duration} seconds"
                    )
                    time.sleep(backoff_duration)

                try:
                    return getitem_fn(self, index)
                except bdb.BdbQuit:
                    raise
                except Exception as e:
                    msg = str(e)
                    err_type = e.__class__.__qualname__

                    if verbosity == 0:
                        pass
                    elif verbosity == 1:
                        source = '.'.join([type(self).__module__, type(self).__qualname__])
                        logging.warning(
                            f"[retry {retry}] Dataloading error: {err_type} - "
                            f"{msg} (from {source})",
                            exc_info=False
                        )
                    else:
                        logging.warning(
                            f"[retry {retry}] {err_type}: {msg}, {do_retry=}",
                            exc_info=True,
                        )

                    if do_retry:
                        index = rng.randint(0, len(self))
                    else:
                        return DataloadFailure(index, msg, err_type)

            raise Exception(f"Dataloading failed after {MAX_RETRIES} retries")

        return getitem_wrapper

    return decorator
