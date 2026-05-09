import random
from collections import defaultdict, deque

from torch.utils.data import Sampler


class PatientAwareBatchSampler(Sampler):
    """
    Round-robin batch sampler: each "round" pulls one sample per patient,
    so consecutive samples (and therefore each batch) cover as many
    distinct patients as possible.

    The sampler reads ``dataset.patient_ids`` lazily inside ``__iter__``
    and ``__len__``, so a dataset that re-samples its negatives between
    epochs (changing ``len(dataset)`` and patient composition) stays in
    sync without rebuilding the sampler.
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = True,
                 shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self._base_seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _interleave(self) -> list:
        rng = random.Random(self._base_seed + self._epoch)

        groups = defaultdict(list)
        for i, p in enumerate(self.dataset.patient_ids):
            groups[p].append(i)

        # deterministic patient order, then shuffle once per epoch
        patients = sorted(groups.keys())
        if self.shuffle:
            for p in patients:
                rng.shuffle(groups[p])
            rng.shuffle(patients)

        queues = [deque(groups[p]) for p in patients]
        merged = []
        while queues:
            next_round = []
            for q in queues:
                merged.append(q.popleft())
                if q:
                    next_round.append(q)
            queues = next_round
        return merged

    def __iter__(self):
        order = self._interleave()
        for i in range(0, len(order), self.batch_size):
            batch = order[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
