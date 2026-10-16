import pickle
import warnings
from src.deepfm.model import DeepFM
from src.deepfm.dataset import get_dataloader
from src.deepfm.trainer import Trainer
from src.utils.parser import config
from deepctr_torch.inputs import SparseFeat, DenseFeat

warnings.filterwarnings("ignore")


def load_data(file_path):
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

    x_train = loaded_data["x_train"]
    y_train = loaded_data["y_train"]
    x_val = loaded_data["x_val"]
    y_val = loaded_data["y_val"]
    x_test = loaded_data["x_test"]
    y_test = loaded_data["y_test"]

    print("\nPreprocessing file: import complete.")

    return x_train, x_val, x_test, y_train, y_val, y_test


def import_model(sparse_features, dense_features, vocab_sizes):
    # 각 범주형 피처의 vocabulary_size를 적절히 설정
    # embedding_dim은 50으로 설정 (조절 가능)

    # Sparse Feature Columns 생성
    sparse_feature_columns = [
        SparseFeat(feat, vocabulary_size=vocab_sizes[feat], embedding_dim=50) for feat
        in sparse_features
    ]

    dense_feature_columns = [
        DenseFeat(feat, dimension=1) for feat in dense_features  # MinMax Scaled 된 값 포함
    ]

    # FM과 DNN에 같은 피처 사용
    linear_feature_columns = dense_feature_columns
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns

    # DeepFM 모델 생성
    model = DeepFM(linear_feature_columns=linear_feature_columns,
                   dnn_feature_columns=dnn_feature_columns, device='cuda:0')

    # model.load_state_dict(torch.load('weights/DeepFM_movie.pth'))

    return model


def main():
    sparse_features = ['userId',
                       'movieId',
]

    dense_features = [
        'genres',
        'release_date',
        'popularity',
        'runtime',
        'revenue']

    vocab_sizes = {'userId': 230000, 'movieId': 23000}
    '''', 'genres': 20, release_date': 13100, 'popularity': 29000,
                   'runtime': 350, 'revenue': 6600'''
    file_path = "data/train_val_test2.pkl"

    config['model'] = import_model(sparse_features, dense_features, vocab_sizes)
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(file_path)
    dataloaders = get_dataloader(config, x_train, x_val, x_test, y_train, y_val, y_test)
    trainer = Trainer(config, dataloaders)
    trainer.train()
    trainer.evaluate()
    trainer.test()


if __name__ == "__main__":
    main()
