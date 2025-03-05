# Built-in imports
import os

# Third party
import numpy as np
from scipy.stats import ttest_1samp, zscore, ttest_rel
import matplotlib
import matplotlib.pyplot as plt


def get_statistical_maps_for_approach(atlas, approach, condition):

    data_folder = os.path.join(
        "Results",
        "Explainable",
        f"Approach_{approach}",
        atlas,
    )

    subjects = os.listdir(os.path.join(data_folder))

    data_fpaths = [os.path.join(data_folder, x, condition, "data.npz") for x in subjects]

    return data_fpaths


def group_statistics(data_fpaths):
    return _group_statistics_t_maps(data_fpaths)


def _group_statistics_t_maps(data_fpaths):

    participants_t_maps = load_all_t_maps(data_fpaths)

    results = ttest_1samp(participants_t_maps, 0, axis=0)
    t_data = results.statistic
    df = results.df
    z_data = zscore(t_data, axis=None, nan_policy="omit")
    p_data = results.pvalue * results.pvalue.size

    return t_data, z_data, p_data, df


def load_all_t_maps(data_fpaths):

    all_data = []

    for fpath in data_fpaths:
        t_map = load_npz_file(fpath)
        if np.isnan(t_map).sum() == 0:
            all_data.append(t_map)

    all_data = np.stack(all_data)
    return all_data


def load_npz_file(fpath, t_map=True):
    all_data = np.load(fpath)

    if t_map:
        return all_data['data']
    else:
        return all_data['trained'], all_data['untrained']


def save_statistic_data(t_map, z_map, p_map, df, p_threshold, output_folder):

    statistic_data_file = os.path.join(
        output_folder,
        "statistics.npz",
    )

    np.savez_compressed(statistic_data_file, t=t_map, z=z_map, p=p_map, df=df, threshold=p_threshold)
    plot_maps(t_map, z_map, p_map, p_threshold, output_folder)


def plot_maps(t_map, z_map, p_map, p_threshold, output_folder):

    matplotlib.use('Agg')
    valid_data_mask = p_map <= p_threshold

    # Save t-map
    t_map[valid_data_mask == 0] = 0

    t_map_file = os.path.join(
        output_folder,
        "t_map",
    )

    _plot_statistic_map(t_map, "t-Map", t_map_file)

    # Save z-map
    z_map[valid_data_mask == 0] = 0

    z_map_file = os.path.join(
        output_folder,
        "z_map",
    )

    _plot_statistic_map(z_map, "z-Map", z_map_file)

    # Save p-map

    p_map[valid_data_mask == 0] = 1000

    p_map_file = os.path.join(
        output_folder,
        "p_map",
    )

    _plot_statistic_map(p_map, "p-Map", p_map_file, colormap="hot_r")


def _plot_statistic_map(data, name, output_file, colormap="bwr"):

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


def make_dir(fpath):

    if not os.path.isdir(fpath):
        os.makedirs(fpath)


if __name__ == '__main__':
    pass
