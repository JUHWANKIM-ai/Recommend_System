import sys
import argparse

sys.argv = ['']
# ---------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--epochs", type=int, default=500, help="epochs")
parser.add_argument("--device", type=str, default='cuda', help="device")
parser.add_argument("--patience", type=int, default=10, help="patience")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size")

opt_train = parser.parse_args()
print(opt_train)
# ---------------------------------------------------------------------------------------------------------------------#
config = dict()
config['train'] = opt_train
