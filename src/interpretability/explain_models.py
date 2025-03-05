# My imports
from .tools import get_all_models_folders, get_trained_untrained_model, get_subjects, get_subject_maps, init_baseline


def model_explanation_approach(atlas):
    baseline = init_baseline(atlas)
    models_folders = get_all_models_folders(atlas)

    for model_folder in models_folders:
        trained_model, untrained_model = get_trained_untrained_model(model_folder, atlas)
        subjects_for_model = get_subjects(model_folder)

        for subject in subjects_for_model:
            get_subject_maps(trained_model, untrained_model, subject, atlas, baseline)


if __name__ == '__main__':
    pass
