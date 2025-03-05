# Built-in imports
import os

import numpy as np

# My imports
from .tools import get_statistical_maps_for_approach, group_statistics, save_statistic_data, make_dir


CONDITIONS = ["Emotion", "Gambling", "Language", "Motor", "Relational", "Social", "WM"]


def explainable_maps_statistics(atlas, approach):

    output_folder = os.path.join(
        "Results",
        "Explainable",
        "Group",
        f"Approach_{approach}",
        atlas,
    )

    for condition in CONDITIONS:
        all_data = get_statistical_maps_for_approach(atlas, approach, condition)

        t_map, z_map, p_map, df = group_statistics_for_condition(all_data)

        condition_folder = os.path.join(
            output_folder,
            condition,
        )

        p_thresh = _p_value_threshold(p_map)
        make_dir(condition_folder)

        save_statistic_data(t_map, z_map, p_map, df, p_thresh, condition_folder)


def group_statistics_for_condition(data_fpaths):

    t_map, z_map, p_map, df = group_statistics(data_fpaths)

    return t_map, z_map, p_map, df


def _p_value_threshold(p_map):

    p_values = p_map.flatten()
    sorter_p_values = np.sort(p_values)

    n_values = sorter_p_values.shape[0]
    idx_threshold = int(n_values * 0.05)

    thresh_p = sorter_p_values[idx_threshold]

    if thresh_p > 0.05:
        thresh_p = 0.05

    return thresh_p


if __name__ == '__main__':
    pass
