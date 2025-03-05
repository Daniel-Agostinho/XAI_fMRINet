# Built-in imports
import os
import glob

# Third-party imports
import numpy as np
from sklearn.model_selection import KFold

# My imports
from .tools import get_available_participants_by_atlas, build_dataset


def cross_val_iterator(atlas, k_folds, shuffle_classes=False):

    # Get all participants
    participants = get_available_participants_by_atlas(atlas)

    # Select 10 % test
    test_k_fold = KFold(n_splits=k_folds, shuffle=True)
    outer_kfold = test_k_fold.split(participants)

    for fold_test, test_fold in enumerate(outer_kfold):

        train_idx, test_idx = test_fold

        test_participants = participants[test_idx]
        train_participants = participants[train_idx]

        test_dataset = build_dataset(test_participants, atlas, shuffle_classes=shuffle_classes)

        train_generator = _inner_validation_fold(train_participants, atlas, shuffle_classes=shuffle_classes)

        yield fold_test, test_participants, test_dataset, train_generator


def _inner_validation_fold(participants, atlas,  shuffle_classes=False):

    valid_k_fold = KFold(n_splits=10, shuffle=True)
    inner_fold = valid_k_fold.split(participants)

    for fold_train, train_fold in enumerate(inner_fold):
        train_idx, validation_idx = train_fold

        train_participants = participants[train_idx]
        validation_participants = participants[validation_idx]

        train_dataset = build_dataset(train_participants, atlas, shuffle_classes=shuffle_classes)
        validation_dataset = build_dataset(validation_participants, atlas, shuffle_classes=shuffle_classes)
        yield fold_train, train_dataset, validation_dataset


if __name__ == '__main__':
    pass
