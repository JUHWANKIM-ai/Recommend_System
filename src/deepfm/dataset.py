import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TrainDataset(Dataset):
    def __init__(self, x_train, y_train):
        # Pandas DataFrame을 numpy 배열로 변환 후 tensor로 변환
        self.x_train = torch.tensor(x_train.values, dtype=torch.float32)
        self.y_train = torch.tensor(y_train.values, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]


def get_dataloader(config, x_train, x_val, x_test, y_train, y_val, y_test):
    print('\nGetting dataloader...')
    train_dataset = TrainDataset(x_train, y_train)
    val_dataset = TrainDataset(x_val, y_val)
    test_dataset = TrainDataset(x_test, y_test)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config['train'].batch_size, shuffle=False),
        'val': DataLoader(val_dataset, batch_size=config['train'].batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=config['train'].batch_size, shuffle=False),
    }
    print('Dataloader: import complete.')
    return dataloaders
