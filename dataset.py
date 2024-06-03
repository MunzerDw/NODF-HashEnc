from torch.utils.data import Dataset


class ObservationPoints(Dataset):

    def __init__(self, coords, signal=None):
        """
        coords: torch.tensor (N x 3) spatial coordinates
        signal: torch.tensor (N x M) M angular samples at each spatial location
        """
        super().__init__()
        self.coords = coords
        self.signal = signal

    def __getitem__(self, idx):
        item = {}

        item["coords"] = self.coords[idx, :]

        if self.signal is not None:
            item["signal"] = self.signal[idx, :]

        return item

    def __len__(self):
        return self.coords.shape[0]
