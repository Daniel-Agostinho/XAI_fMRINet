# Built-in imports
import os
import glob
import pickle


# Third-party imports
import numpy as np


# My imports
from .datasets import CNNDataset


def get_available_participants_by_atlas(atlas):

    fpaths = glob.glob(os.path.join(
        "Dataset",
        "Data",
        atlas,
        "*.pkl"
    ))

    participants = list(map(lambda x: os.path.split(x)[-1].split(".")[0], fpaths))

    return np.asarray(participants)


def build_dataset(participants, atlas, shuffle_classes=False):

    data, labels = None, None

    for participant in participants:

        participant_data, participant_labels = get_participant_data(participant, atlas, shuffle_classes=shuffle_classes)

        if data is None:
            data = participant_data
            labels = participant_labels
        else:
            data = np.concatenate((data, participant_data), axis=0)
            labels = np.concatenate((labels, participant_labels), axis=0)

    data, labels = _data_to_pytorch_format(data, labels)

    return CNNDataset(data, labels, transform=True)


def _save_shuffle_data(participant, atlas, data, labels):

    output_folder = os.path.join(
        "Dataset",
        "Shuffle",
        atlas,
    )

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    file_name = os.path.join(
        output_folder,
        f"{participant}.npz",
    )

    np.savez_compressed(file_name, data=data, labels=labels)


def get_participant_data(participant_id, atlas, shuffle_classes=False):

    shuffle_data_fpath = os.path.join(
        "Dataset",
        "Shuffle",
        atlas,
        f"{participant_id}.npz",
    )

    if os.path.isfile(shuffle_data_fpath) and shuffle_classes:
        all_data = np.load(shuffle_data_fpath)
        return all_data['data'], all_data['labels']

    data_fpath = os.path.join(
        "Dataset",
        "Data",
        atlas,
        f"{participant_id}.pkl",
    )

    raw_data, labels = load_pkl_data(data_fpath)
    n_blocks = 21  # 15 seconds of data

    final_data = []
    final_labels = []

    for entry, label in zip(raw_data, labels):
        temp_data = np.asarray(entry)
        data_in = temp_data[:, 0:n_blocks]

        if data_in.shape[1] == n_blocks:
            final_data.append(data_in)
            final_labels.append(label)

    participant_data, participant_labels = np.asarray(final_data), np.asarray(final_labels)

    if shuffle_classes:
        np.random.shuffle(participant_labels)
        _save_shuffle_data(participant_id, atlas, participant_data, participant_labels)

    return participant_data, participant_labels


def load_pkl_data(fpath):

    with open(fpath, "rb") as file:
        all_data = pickle.load(file)

    return all_data['data'], all_data['labels']


def _data_to_pytorch_format(data, labels):

    data = __reshape_pytorch(data)
    labels = __convert_labels(labels)
    return data, labels


def __convert_labels(labels):
    new_labels = np.zeros((labels.shape[0], 7))

    for i in range(labels.shape[0]):
        new_labels[i, int(labels[i])] = 1

    return new_labels


def __reshape_pytorch(data):
    new_x = data.reshape((
        data.shape[0],
        1,
        data.shape[1],
        data.shape[2]
    ))

    return new_x


if __name__ == '__main__':
    pass
