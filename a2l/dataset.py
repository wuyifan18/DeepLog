# dataset.py

import os
import torch


class OpenStackDataset(torch.utils.data.Dataset):
    def __init__(self, data, testing=False):
        super(OpenStackDataset, self).__init__()
        self.testing=testing

        # read data
        self.inputs, self.labels = self._read_data(data, testing=testing)

    def _read_data(self, data):
        return None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.testing:
            return self.inputs[idx]
        else:
            return self.inputs[idx], self.labels[idx]

class LogDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size):
        super(LogDataset, self).__init__()
        self.window_size = window_size

        # read data
        self.inputs, self.labels = self._read_data(data)

    def _read_data(self, data):
        # Function to read HFDS data
        # Args:
        #   - data: str
        #       Path to data
        #   - testing: bool
        #       Flag to indicate if data is a testing dataset
        #       If a testing dataset, return inputs only.

        # TODO: handle testing datasets when testing=True
        assert os.path.exists(data), 'Provided data {} does not exists'.format(data)

        num_sessions = 0
        inputs, labels = [], []
        with open(data, 'r') as file:
            for line in file.readlines():
                num_sessions += 1
                line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
                for i in range(len(line) - self.window_size):
                    inputs.append(line[i:i + self.window_size])
                    labels.append(line[i + self.window_size])

        # print dataset stats
        self.num_session = num_sessions
        self.num_seq = len(inputs)
        print('Printing dataset statistics: {}'.format(data))
        print('Number of sessions({}): {}'.format(data, self.num_session))
        print('Number of seqs({}): {}'.format(data, self.num_seq))

        # convert inputs and labels to tensors
        inputs = torch.tensor(inputs, dtype=torch.float)
        labels = torch.tensor(labels)

        return inputs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]