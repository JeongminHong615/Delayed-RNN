from dataset.delaysequence import DelaySequenceDataset
from dataset.delaysequence_hard import DelaySequenceHardDataset

DATASET_REGISTRY = {
    "delaysequence": DelaySequenceDataset,
    "delaysequence_hard": DelaySequenceHardDataset,
}