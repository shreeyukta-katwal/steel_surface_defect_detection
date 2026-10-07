import torch


class EarlyStopping:

    def __init__(self,
                 patience=5,
                 min_delta=0.0):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")

        self.counter = 0

        self.stop = False

    def __call__(self, val_loss):

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.stop = True


class CheckpointSaver:

    def __init__(self, path):

        self.best_loss = float("inf")

        self.path = path

    def save(self, model, val_loss):

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            torch.save(
                model.state_dict(),
                self.path
            )

            print("Best model saved.")
