# Third party imports
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim


class CNNTrainner:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataset: Dataset,
            valid_dataset: Dataset,
            device: int,
            learning_rate=1e-3,
            max_epochs=250,
            early_stop_limit=80,
            batch_size=246,
    ):

        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_loader, self.valid_loader = self.initialize_data_loaders(train_dataset, valid_dataset,
                                                                            batch_size, device)
        self.max_epochs = max_epochs
        self.early_stop = early_stop_limit
        self.device = device
        self.training = True

    def train(self, logger):
        unbalance_weights = self.imbalance_weights(self.train_loader.dataset.labels, self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=unbalance_weights)

        best_valid_loss = 1000
        early_stop_count = 0
        current_epoch = 1

        while self.training:
            train_loss, valid_loss = self._run_epoch(current_epoch, criterion)
            logger.log_train([current_epoch, train_loss, valid_loss])
            current_epoch += 1

            # Store best model if performance improved
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                logger.save_model(self.model.state_dict())
                early_stop_count = 0

            # Otherwise increase early stop counter
            else:
                early_stop_count += 1

            self.training = self._stop_train(early_stop_count, current_epoch)

        return best_valid_loss

    def _run_epoch(self, epoch, criterion):

        train_loss = 0
        validation_loss = 0

        # Train
        self.model.train()
        for samples, t_data in enumerate(self.train_loader):

            train_data, train_label = t_data

            # Clear the gradients
            self.optimizer.zero_grad()

            # Forward Pass
            target = self.model(train_data)

            # Find the Loss
            loss = criterion(target, train_label)

            # Calculate gradients
            loss.backward()

            # Update Weights
            self.optimizer.step()

            # Calculate Loss
            train_loss += loss.item()

        mean_train_loss = train_loss / len(self.train_loader)

        # Validation
        self.model.eval()
        with torch.no_grad():
            for valid_data, valid_labels in self.valid_loader:
                # Forward Pass
                val_prediction = self.model(valid_data)

                # Find the Loss
                loss = criterion(val_prediction, valid_labels)

                # Calculate Loss
                validation_loss += loss.item()

            mean_valid_loss = validation_loss / len(self.valid_loader)

        print(f"----- Epoch {epoch} summary -----\n")
        print(f"Training loss: {mean_train_loss: .5f}\n")
        print(f"Validation loss: {mean_valid_loss: .5f}\n\n")

        return mean_train_loss, mean_valid_loss

    def _stop_train(self, early_count, epoch):
        if early_count == self.early_stop:
            return False

        elif epoch >= self.max_epochs:
            return False

        return True

    @staticmethod
    def initialize_data_loaders(train, valid, batch_size, device):
        train.set_device(device)
        valid.set_device(device)

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)
        return train_loader, valid_loader

    @staticmethod
    def imbalance_weights(labels, device):
        labels = torch.argmax(labels, dim=1).to('cpu')
        number_of_classes = labels.max() + 1
        weights = torch.zeros(number_of_classes).to(device, dtype=torch.float)
        total_number_of_samples = labels.size(dim=0)

        for i in range(number_of_classes):
            class_i = labels == i
            samples_of_class_i = class_i.sum()
            weights[i] = 1 - (samples_of_class_i / total_number_of_samples)

        return weights


if __name__ == '__main__':
    pass
