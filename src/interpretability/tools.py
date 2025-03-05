# Built-in imports
import os
import pickle

# Third party
import torch
import pandas as pd
import numpy as np
from captum.attr import DeepLiftShap
from scipy.stats import ttest_rel, ttest_1samp, zscore, ttest_ind
import matplotlib
import matplotlib.pyplot as plt

# My imports
from src.classification.networks.cnn import FMRINet


CUDA = 0

ATLAS_ROIS = {
    "BN": 246,
    "HCP": 426,
}


TARGET_NAMES = {
    0: "Emotion",
    1: "Gambling",
    2: "Language",
    3: "Motor",
    4: "Relational",
    5: "Social",
    6: "WM",
}


def get_subject_maps(trained_model, untrained_model, subject, atlas, baseline):

    subject_data, subject_labels = load_subject_data(subject, atlas)
    subject_labels = torch.from_numpy(subject_labels)
    targets = np.unique(subject_labels)
    samples, _, _ = subject_data.shape
    target_baseline = np.asarray([baseline for x in range(int(samples))])

    print(f"----- {subject} -----\n")
    trained_maps, untrained_maps = layer_wise_importance(trained_model, untrained_model, subject_data, subject_labels,
                                                         target_baseline)

    for target in targets:
        print(f"Condition: {target}")
        trained_data, untrained_data = trained_maps[subject_labels == target, ...], untrained_maps[subject_labels == target, ...]
        label_t_map, label_p_map = statistical_analysis(trained_data, untrained_data)
        save_data(1, label_t_map, label_p_map, trained_data, untrained_data, atlas,
                  participant=subject, target=int(target))


def layer_wise_importance(trained_model, untrained_model, data, label, baseline):

    data, baseline = transform_data_for_cnn(data, baseline)
    data = data.to(device=f"cuda:{CUDA}")
    baseline = baseline.to(device=f"cuda:{CUDA}")

    if not isinstance(label, int):
        label = label.to(device=f"cuda:{CUDA}")

    # Train model
    trained_model.to(f"cuda:{CUDA}")
    trained_model.eval()

    trained_importance = get_model_importance(trained_model, data, baseline, label)

    trained_model.cpu()

    if untrained_model is not None:
        # Untrained model

        untrained_model.to(f"cuda:{CUDA}")
        untrained_model.eval()

        untrained_importance = get_model_importance(untrained_model, data, baseline, label)

        untrained_model.cpu()

        return trained_importance, untrained_importance

    return trained_importance


def get_all_models_folders(atlas, shuffle_models=False):
    trained_atlas_folder = os.path.join(
        "Results",
        "Classification",
        "fMRINet",
        "Trained",
        atlas,
    )

    untrained_atlas_folder = os.path.join(
        "Results",
        "Classification",
        "fMRINet",
        "Shuffle",
        atlas,
    )

    models_folders = [os.path.join(trained_atlas_folder, x) for x in os.listdir(trained_atlas_folder)]

    if shuffle_models:
        shuffle_models_folders = [os.path.join(untrained_atlas_folder, x) for x in os.listdir(untrained_atlas_folder)]
        return models_folders, shuffle_models_folders

    return models_folders


def get_trained_untrained_model(model_fpath, atlas, shuffle=False):

    model_trained_fpath = os.path.join(
        model_fpath,
        "test_model"
    )

    network_parameters = {
        "f1": 8,
        "froi": ATLAS_ROIS[atlas],
        "f2": 16,
        "d": 2,
        "p": 0.2,
        "n_classes": 7
    }

    model = FMRINet(**network_parameters)
    model.load_state_dict(torch.load(model_trained_fpath, weights_only=True))

    if shuffle:
        return model

    untrained_model = FMRINet(**network_parameters)

    return model, untrained_model


def get_subjects(model_fpath):

    subjects_list = os.path.join(
        model_fpath,
        "subjects.txt",
    )

    all_subjects_table = pd.read_table(subjects_list, header=None)
    all_subjects = all_subjects_table[0].to_list()
    return all_subjects


def load_subject_data(subject, atlas, shuffle=False):

    if shuffle:
        data_fpath = os.path.join(
            "Dataset",
            "Shuffle",
            atlas,
            f"{subject}.npz",
        )

        all_data = np.load(data_fpath)

        return all_data["data"], all_data["labels"]

    data_fpath = os.path.join(
        "Dataset",
        "Data",
        atlas,
        f"{subject}.pkl",
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

    return np.asarray(final_data), np.asarray(final_labels)


def load_multiple_subjects_data(subjects, atlas):

    data, labels = None, None

    for subject in subjects:
        subject_data, subject_labels = load_subject_data(subject, atlas)

        if data is None:
            data, labels = subject_data, subject_labels
        else:
            data = np.concatenate((data, subject_data), axis=0)
            labels = np.concatenate((labels, subject_labels), axis=0)

    return data, labels


def load_pkl_data(fpath):

    with open(fpath, "rb") as file:
        all_data = pickle.load(file)

    return all_data['data'], all_data['labels']


def init_baseline(atlas):
    f_rois = ATLAS_ROIS[atlas]

    baseline = np.abs(np.random.normal(loc=0, scale=0.2, size=(1, f_rois, 21)))

    return baseline


def transform_data_for_cnn(data, baseline):
    samples, rois, time = data.shape

    if samples == 1:
        data = data.reshape(1, 1, rois, time)
        baseline = baseline.reshape(1, 1, rois, time)
        data = np.concatenate((data, data), axis=0)
        baseline = np.concatenate((baseline, baseline), axis=0)

    else:
        data = data.reshape(samples, 1, rois, time)
        baseline = baseline.reshape(samples, 1, rois, time)

    baseline = torch.from_numpy(baseline)
    data = torch.from_numpy(data)

    return data.to(dtype=torch.float), baseline.to(dtype=torch.float)


def get_model_importance(model, input_data, baseline, target):

    deep_lift = DeepLiftShap(model)
    attribution = deep_lift.attribute(input_data, baseline, target=target)
    attribution = attribution.cpu().detach().numpy()

    return attribution


def statistical_analysis(trained_importance, untrained_importance, paired=True):

    if paired:
        t_test = ttest_rel(trained_importance, untrained_importance, axis=0)
    else:
        t_test = ttest_ind(trained_importance, untrained_importance, axis=0)

    t_values = t_test.statistic.squeeze()
    p_value = t_test.pvalue.squeeze() * t_test.pvalue.squeeze().size

    return t_values, p_value


def save_data(approach, t_map, p_map, trained_maps, untrained_maps, atlas, participant=None, target=None, model_id=None):

    main_output_folder = os.path.join(
        "Results",
        "Explainable",
        f"Approach_{approach}",
        atlas,
    )

    condition = TARGET_NAMES[target]

    if approach == 1 or approach == 3 or approach == 4:
        output_folder = os.path.join(
            main_output_folder,
            f"{participant}",
            condition,
        )

    elif approach == 2:
        output_folder = os.path.join(
            main_output_folder,
            f"Model_{model_id}",
            condition,
        )

    else:
        raise ValueError("Invalid approach!")

    make_dir(output_folder)

    data_file = os.path.join(
        output_folder,
        "data.npz",
    )

    np.savez_compressed(data_file, data=t_map, pvalue=p_map, trained=trained_maps, untrained=untrained_maps)

    # Plot
    significant_t = t_map
    significant_t[p_map > 0.01] = 0

    plot_data(significant_t, participant, output_folder, file_name="t_map")
    plot_data(trained_maps, participant, output_folder, file_name="trained_importance")
    plot_data(untrained_maps, participant, output_folder, file_name="untrained_importance")


def plot_data(data, participant, output_folder, file_name="Map"):
    matplotlib.use('Agg')

    if file_name != "t_map":
        data = data.mean(axis=0)
        data = np.squeeze(data)

    map_folder = os.path.join(
        output_folder,
        "Maps",
    )

    make_dir(map_folder)

    map_fpath = os.path.join(
        map_folder,
        file_name,
    )

    max_value = np.abs(data).max()

    fig = plt.figure()
    plt.imshow(data, aspect="auto", cmap="bwr", vmin=-max_value, vmax=max_value)
    plt.colorbar()
    plt.title(f"{participant}")
    plt.xlabel("TR")
    plt.ylabel("ROIs")
    plt.tight_layout()
    plt.savefig(map_fpath)
    plt.close(fig)


def make_dir(fpath):

    if not os.path.isdir(fpath):
        os.makedirs(fpath)


def get_model_maps(model_id, trained_model, untrained_model, subjects, atlas, baseline):

    print(f"Analyzing Model {model_id} ....")
    data, labels = load_multiple_subjects_data(subjects, atlas)

    samples, _, _ = data.shape
    idx_i, idx_f, batch = 0, 50, 50

    trained_maps, untrained_maps = None, None

    while idx_i < samples:
        batch_input = data[idx_i:idx_f, ...]
        batch_target = torch.from_numpy(labels[idx_i:idx_f, ...])
        target_baseline = np.asarray([baseline for x in range(batch_input.shape[0])])
        temp_trained, temp_untrained = layer_wise_importance(trained_model, untrained_model, batch_input,
                                                             batch_target, target_baseline)

        if trained_maps is None:
            trained_maps = temp_trained
            untrained_maps = temp_untrained
        else:
            trained_maps = np.concatenate((trained_maps, temp_trained), axis=0)
            untrained_maps = np.concatenate((untrained_maps, temp_untrained), axis=0)

        idx_i += batch
        idx_f += batch

    assert trained_maps.shape[0] == samples
    targets = np.unique(labels)

    for target in targets:
        print(f"Extracting t-maps: {TARGET_NAMES[target]}")
        target_trained, target_untrained = trained_maps[labels == target, ...], untrained_maps[labels == target, ...]
        label_t_map, label_p_map = statistical_analysis(target_trained, target_untrained)
        save_data(2, label_t_map, label_p_map, target_trained, target_untrained, atlas,
                  model_id=model_id, target=int(target))


def save_temp_data(temp_folder, participant, data, target, shuffle, labels=None, approach_4=False):

    if approach_4:
        output_folder = os.path.join(
            temp_folder,
            f"{participant}",
        )

    else:
        output_folder = os.path.join(
            temp_folder,
            f"{participant}",
            TARGET_NAMES[target],
        )

    make_dir(output_folder)

    file_name = os.path.join(
        output_folder,
        f"{'trained' if not shuffle else 'shuffle'}.npz",
    )

    if approach_4:
        np.savez_compressed(file_name, data=data, labels=labels)
    else:
        np.savez_compressed(file_name, data=data)


def extract_model_subject_importance(model_folder, atlas, baseline, temp_folder, shuffle=False, approach_4=False):

    subjects_for_model = get_subjects(model_folder)
    model = get_trained_untrained_model(model_folder, atlas, shuffle=True)

    for subject in subjects_for_model:

        # Load subject data
        subject_data, subject_labels = load_subject_data(subject, atlas, shuffle=shuffle)
        torch_labels = torch.from_numpy(subject_labels)

        # Build baseline for samples
        samples, _, _ = subject_data.shape
        subject_baselines = np.asarray([baseline for x in range(int(samples))])

        # Extract importance for samples
        print(f"----- {subject} -----\n")
        importance = layer_wise_importance(model, None, subject_data, torch_labels, subject_baselines)

        # Approach_4
        if approach_4:
            save_temp_data(temp_folder, subject, importance, None, shuffle, labels=subject_labels, approach_4=True)

        else:
            # Save importance by class
            targets = np.unique(subject_labels)

            for target in targets:
                target_importance = importance[subject_labels == target, ...]
                save_temp_data(temp_folder, subject, target_importance, target, shuffle)


def approach_3_statistics(temp_folder, atlas):

    subjects = [x for x in os.listdir(os.path.join(temp_folder))]

    for subject in subjects:
        _perform_subject_statistics_for_all_conditions(subject, temp_folder, atlas, paired=False)


def _perform_subject_statistics_for_all_conditions(subject, data_folder, atlas, paired=True):

    conditions = ["Emotion", "Gambling", "Language", "Motor", "Relational", "Social", "WM"]

    for condition, _ in enumerate(conditions):
        __get_condition_statistics(data_folder, subject, condition, atlas, paired=paired)


def __get_condition_statistics(main_folder, subject, condition, atlas, paired=True):

    data_folder = os.path.join(
        main_folder,
        f"{subject}",
        TARGET_NAMES[condition],
    )

    trained_data = np.load(os.path.join(data_folder, "trained.npz"))['data']
    shuffle_data = np.load(os.path.join(data_folder, "shuffle.npz"))['data']

    t_maps, p_maps = statistical_analysis(trained_data, shuffle_data, paired=paired)
    save_data(3, t_maps, p_maps, trained_data, shuffle_data, participant=subject, atlas=atlas, target=condition)


def approach_4_statistics(temp_folder, atlas):

    subjects = [x for x in os.listdir(os.path.join(temp_folder))]

    for subject in subjects:
        _perform_approach_4_subject_statistics(subject, temp_folder, atlas)


def _perform_approach_4_subject_statistics(subject, temp_folder, atlas):

    data_folder = os.path.join(
        temp_folder,
        f"{subject}",
    )

    trained_info = np.load(os.path.join(data_folder, "trained.npz"))
    trained_data, labels = trained_info['data'], trained_info['labels']

    shuffle_data = np.load(os.path.join(data_folder, "shuffle.npz"))['data']

    conditions = np.unique(labels)

    for condition in conditions:
        condition_trained_data = trained_data[labels == condition, ...]
        condition_shuffle_data = shuffle_data[labels == condition, ...]
        t_maps, p_maps = statistical_analysis(condition_trained_data, condition_shuffle_data)
        save_data(4, t_maps, p_maps, condition_trained_data, condition_shuffle_data, participant=subject, atlas=atlas,
                  target=condition)


def extract_model_importance(model_folder, atlas, baseline, temp_folder, shuffle=False, subject_store=False):

    subjects_for_model = get_subjects(model_folder)
    model = get_trained_untrained_model(model_folder, atlas, shuffle=True)

    importance_for_model = None
    labels_for_model = None

    for subject in subjects_for_model:
        # Load subject data
        subject_data, subject_labels = load_subject_data(subject, atlas, shuffle=shuffle)
        torch_label = torch.from_numpy(subject_labels)

        # Build baseline for samples
        samples, _, _ = subject_data.shape
        target_baselines = np.asarray([baseline for x in range(int(samples))])

        # Extract importance for samples
        subject_importance = layer_wise_importance(model, None, subject_data, torch_label, target_baselines)

        if subject_store:
            print(f"-----{subject}-----")
            targets = np.unique(subject_labels)
            for target in targets:
                target_data = subject_importance[subject_labels == target, ...]
                target_data = np.squeeze(target_data)

                if len(target_data.shape) > 2:
                    target_data = target_data.mean(axis=0)

                save_temp_data(temp_folder, subject, target_data, target, shuffle)

        else:
            if importance_for_model is None:
                importance_for_model = subject_importance
                labels_for_model = subject_labels
            else:
                importance_for_model = np.concatenate((importance_for_model, subject_importance), axis=0)
                labels_for_model = np.concatenate((labels_for_model, subject_labels), axis=0)

    if not subject_store:
        targets = np.unique(labels_for_model)

        for target in targets:
            print(f"{TARGET_NAMES[target]}")
            target_data = importance_for_model[labels_for_model == target, ...]
            save_temp_data_approach5(temp_folder, target_data, target, shuffle)


def save_temp_data_approach5(temp_folder, data, target, shuffle):

    output_folder = os.path.join(
        temp_folder,
        TARGET_NAMES[target],
    )

    make_dir(output_folder)

    file_name = os.path.join(
        output_folder,
        f"{'trained' if not shuffle else 'shuffle'}.npz",
    )

    if os.path.isfile(file_name):
        old_data = np.load(file_name)['data']
        new_data = np.concatenate((old_data, data), axis=0)
        np.savez_compressed(file_name, data=new_data)
    else:
        np.savez_compressed(file_name, data=data)


def approach_5_statistics(temp_folder, atlas):

    conditions = [x for x in os.listdir(temp_folder)]

    for condition in conditions:
        print(condition)

        condition_folder = os.path.join(
            temp_folder,
            condition,
        )

        trained_data = np.load(os.path.join(condition_folder, "trained.npz"))['data']
        shuffle_data = np.load(os.path.join(condition_folder, "shuffle.npz"))['data']

        t_map, p_map = statistical_analysis(trained_data, shuffle_data, paired=False)
        z_map = zscore(t_map, axis=None)

        _save_approach_data(5, t_map, z_map, p_map, condition, atlas)


def approach_6_statistics(temp_folder, atlas):

    # temp, participant, condition
    subjects = [x for x in os.listdir(temp_folder)]

    conditions = os.listdir(os.path.join(temp_folder, subjects[0]))

    for condition in conditions:

        trained_data, shuffle_data = __get_condition_data(subjects, temp_folder, condition)

        t_map, p_map = statistical_analysis(trained_data, shuffle_data, paired=True)
        z_map = zscore(t_map, axis=None)

        _save_approach_data(6, t_map, z_map, p_map, condition, atlas, trained=trained_data, shuffle=shuffle_data)


def __get_condition_data(subjects, data_folder, condition):

    condition_trained_data = []
    condition_shuffled_data = []

    for subject in subjects:

        subject_condition_folder = os.path.join(
            data_folder,
            f"{subject}",
            condition,
        )

        trained_data = np.load(os.path.join(subject_condition_folder, "trained.npz"))['data']
        shuffle_data = np.load(os.path.join(subject_condition_folder, "shuffle.npz"))['data']

        condition_trained_data.append(trained_data)
        condition_shuffled_data.append(shuffle_data)

    return np.stack(condition_trained_data), np.stack(condition_shuffled_data)


def _save_approach_data(approach, t, z, p, condition, atlas, trained=None, shuffle=None):

    output_folder = os.path.join(
        "Results",
        "Explainable",
        f"Approach_{approach}",
        atlas,
        condition,
    )

    make_dir(output_folder)

    file_name = os.path.join(
        output_folder,
        "statistics.npz",
    )

    if trained is not None:
        np.savez_compressed(file_name, t=t, z=z, p=p, trained=trained, shuffle=shuffle)

    np.savez_compressed(file_name, t=t, z=z, p=p)

    # Plot
    _plot_maps(t, z, p, output_folder)


def _plot_maps(t_map, z_map, p_map, output_folder):

    matplotlib.use('Agg')
    valid_data_mask = p_map <= 0.05

    # Save t-map
    t_map[valid_data_mask == 0] = 0

    t_map_file = os.path.join(
        output_folder,
        "t_map",
    )

    __plot_statistic_map(t_map, "t-Map", t_map_file)

    # Save z-map
    z_map[valid_data_mask == 0] = 0

    z_map_file = os.path.join(
        output_folder,
        "z_map",
    )

    __plot_statistic_map(z_map, "z-Map", z_map_file)

    # Save p-map

    p_map[valid_data_mask == 0] = 1000

    p_map_file = os.path.join(
        output_folder,
        "p_map",
    )

    __plot_statistic_map(p_map, "p-Map", p_map_file, colormap="hot_r")


def __plot_statistic_map(data, name, output_file, colormap="bwr"):

    if name != "p-Map":
        max_value = np.abs(data).max()
        min_value = -max_value

    else:
        max_value = 1
        min_value = 0

    fig = plt.figure()
    plt.imshow(data, aspect="auto", cmap=colormap, vmin=min_value, vmax=max_value)
    plt.colorbar()
    plt.title(f"{name}")
    plt.xlabel("TR")
    plt.ylabel("ROIs")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)


if __name__ == '__main__':
    pass
