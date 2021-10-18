from NT import data_process, batch_size, print_acc, CNN
import torch.utils.data as Data
from tqdm import tqdm, trange
import torch


def predict():
    dataset, _ = data_process('predict')
    data_iter = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
        pin_memory=True,
        drop_last=False
    )
    tqdm.write('predicting')
    cnn = torch.load('cnn_all.pt')
    cnn.to('cpu')
    cnn.eval()
    correct, total = torch.zeros(11), torch.zeros(11)
    with torch.no_grad():
        t = trange(len(data_iter))
        for X, y in data_iter:
            t.update(1)
            X = X.float()
            y_hat1 = cnn(X)
            y_hat1_max = y_hat1.argmax(dim=1)
            res = (y_hat1_max == y[:, 1] + 0).float()
            for index in range(len(y)):
                label_single = y[index, 1]
                correct[label_single] += res[index].item()
                total[label_single] += 1
    t.close()
    print_acc('cnn', correct, total, 11)


if __name__ == '__main__':
    predict()
