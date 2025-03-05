# Built-in imports
import os

# Third party imports
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

LABELS = [0, 1, 2, 3, 4, 5, 6]
TARGET_NAMES = ["Emotion", "Gambling", "Language", "Motor", "Relational", "Social", "WM"]


def make_dir(fpath):
    if not os.path.isdir(fpath):
        os.makedirs(fpath)


def freeze_layers(model):
    all_layers = model.children()
    for i in range(model.number_of_layers - 1):
        layer2freeze = next(all_layers)
        for parameter in layer2freeze.parameters():
            parameter.requires_grad = False


def unfreeze_layers(model):
    all_layers = model.children()
    for i in range(model.number_of_layers - 1):
        layer2freeze = next(all_layers)
        for parameter in layer2freeze.parameters():
            parameter.requires_grad = True


def test_model(model, test_dataset, logger):

    data, labels = test_dataset.data, test_dataset.labels
    model.eval()

    with torch.no_grad():
        prediction = model(data)
        cm, summary, summary_data, disp = _performance_metrics(prediction, labels)

    logger.log_test(model, cm, summary, summary_data, disp)


def _performance_metrics(prediction, true_labels):

    predict = torch.argmax(prediction, dim=1).detach().cpu().numpy()
    true_label = torch.argmax(true_labels, dim=1).detach().cpu().numpy()

    cm = confusion_matrix(true_label, predict)
    summary = classification_report(true_label, predict, labels=LABELS, target_names=TARGET_NAMES)
    summary_data = classification_report(true_label, predict, labels=LABELS, target_names=TARGET_NAMES,
                                         output_dict=True)
    disp = ConfusionMatrixDisplay(cm, display_labels=TARGET_NAMES)
    return cm, summary, summary_data, disp


if __name__ == '__main__':
    pass
