# Third party imports
import torch

# My imports
from src.classification.tools import initialize_model
from src.classification.data_iterators.iterators import cross_val_iterator
from src.classification.networks.trainners import CNNTrainner
from src.classification.networks.loggers import HCPLogger
from src.classification.networks.tools import test_model

ATLAS_ROIS = {
    "BN": 246,
    "HCP": 426,
}

SEED = 1111


def main():
    atlas = "BN"
    k_folds = 10

    trainner_parameters = {
        "max_epochs": 400,
        "early_stop_limit": 80,
        "learning_rate": 1e-3,
        "batch_size": 246,
        "device": 0,
    }

    network_name = "fMRINet"

    network_parameters = {
        "f1": 8,
        "froi": ATLAS_ROIS[atlas],
        "f2": 16,
        "d": 2,
        "p": 0.2,
        "n_classes": 7
    }

    experiment(network_name, atlas, k_folds, network_parameters, trainner_parameters)


def experiment(network_name, atlas, k_folds, network_parameters, trainner_parameters):

    data_iter = cross_val_iterator(atlas, k_folds, shuffle_classes=True)

    for fold_test, test_participants, test_dataset, train_generator in data_iter:
        logger = HCPLogger(fold_test, test_participants, network_name, SEED, atlas)
        model = initialize_model(network_name, network_parameters)

        best_cross_fold = -1
        best_cross_loss = 1000

        for train_fold, train_dataset, validation_dataset in train_generator:
            logger.start_train_log(train_fold)

            trainner = CNNTrainner(model=model, train_dataset=train_dataset, valid_dataset=validation_dataset,
                                   **trainner_parameters)
            inner_fold_loss = trainner.train(logger)

            if inner_fold_loss < best_cross_loss:
                best_cross_loss = inner_fold_loss
                best_cross_fold = train_fold

        # Test
        best_model_fpath = logger.get_best_model(best_cross_fold)

        model = initialize_model(network_name, network_parameters)
        model.load_state_dict(torch.load(best_model_fpath))

        test_model(model, test_dataset, logger)


if __name__ == '__main__':
    main()
