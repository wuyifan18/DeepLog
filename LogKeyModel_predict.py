import torch
import torch.nn as nn
import time

# Device configuration
device = torch.device("cpu")
# Hyperparameters
window_size = 10
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 28
num_candidates = 9
model_path = 'model/Adam with batch_size=2048;epoch=300.pt'


def generate(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    # hdfs = set()
    hdfs = []
    with open('data/' + name, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            line = line + [-1] * (window_size + 1 - len(line))
            # hdfs.add(tuple(line))
            hdfs.append(tuple(line))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print('model_path: {}'.format(model_path))
test_normal_loader = generate('hdfs_test_normal')
test_abnormal_loader = generate('hdfs_test_abnormal')
TP = 0
FP = 0
# Test the model
start_time = time.time()
with torch.no_grad():
    for line in test_normal_loader:
        for i in range(len(line) - window_size):
            seq = line[i:i + window_size]
            label = line[i + window_size]
            seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(label).view(-1).to(device)
            output = model(seq)
            predicted = torch.argsort(output, 1)[0][-num_candidates:]
            if label not in predicted:
                FP += 1
                break
with torch.no_grad():
    for line in test_abnormal_loader:
        for i in range(len(line) - window_size):
            seq = line[i:i + window_size]
            label = line[i + window_size]
            seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(label).view(-1).to(device)
            output = model(seq)
            predicted = torch.argsort(output, 1)[0][-num_candidates:]
            if label not in predicted:
                TP += 1
                break

# Compute precision, recall and F1-measure
FN = len(test_abnormal_loader) - TP
P = 100 * TP / (TP + FP)
R = 100 * TP / (TP + FN)
F1 = 2 * P * R / (P + R)
print(
    'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
        FP,
        FN,
        P,
        R,
        F1))
print('Finished Predicting')
elapsed_time = time.time() - start_time
print('elapsed_time: {}'.format(elapsed_time))
