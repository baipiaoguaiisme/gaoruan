from sklearn import metrics
from tqdm import tqdm
import torch
import numpy as np

from load_data import getLoader


def run_epoch(max_problem, pro_path, skill_path, batch_size, is_train, min_problem_num, max_problem_num,
              model, optimizer, criterion, device, grad_clip):
    total_correct = 0
    total_num = 0
    total_loss = []

    labels = []
    outputs = []

    if is_train:
        model.train()
    else:
        model.eval()

    data_loader = getLoader(max_problem, pro_path, skill_path, batch_size, is_train, min_problem_num, max_problem_num)

    for i, data in tqdm(enumerate(data_loader), desc='加载中...'):

        # shape全都是[80, 199]
        last_problem, last_skill, last_ans, next_problem, next_skill, next_ans, mask = data

        if is_train:
            predict = model(last_problem, last_skill, last_ans, next_problem, next_skill,
                            next_ans)  # torch.Size([80, 199])
            # print(predict)
            # print(predict.shape)
            next_predict = torch.masked_select(predict, mask)  # torch.Size([5114])
            # print(next_predict)
            # print(next_predict.shape)
            next_true = torch.masked_select(next_ans, mask)  # torch.Size([5114])
            # print(next_true)
            # print(next_true.shape)
            loss = criterion(next_predict, next_true)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            labels.extend(next_true.view(-1).data.cpu().numpy())
            outputs.extend(next_predict.view(-1).data.cpu().numpy())

            total_loss.append(loss.item())
            total_num += len(next_true)

            to_pred = (next_predict >= 0.5).long()
            total_correct += (next_true == to_pred).sum()
        else:
            with torch.no_grad():
                predict = model(last_problem, last_skill, last_ans, next_problem, next_skill, next_ans)

                next_predict = torch.masked_select(predict, mask)
                next_true = torch.masked_select(next_ans, mask)

                loss = criterion(next_predict, next_true)

                labels.extend(next_true.view(-1).data.cpu().numpy())
                outputs.extend(next_predict.view(-1).data.cpu().numpy())

                total_loss.append(loss.item())
                total_num += len(next_true)

                to_pred = (next_predict >= 0.5).long()
                total_correct += (next_true == to_pred).sum()

    avg_loss = np.average(total_loss)
    acc = total_correct * 1.0 / total_num
    auc = metrics.roc_auc_score(labels, outputs)
    return avg_loss, acc, auc


if __name__ == '__main__':
    from model import ReKT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReKT(168, 167, 128, 0.4).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    train_loss, train_acc, train_auc = run_epoch(168, r'./data/ASSIST09/train_question.txt',
                                                 r'./data/ASSIST09/train_skill.txt',
                                                 80,
                                                 True, 3, 200, model=model, optimizer=optimizer, criterion=criterion,
                                                 device=device,
                                                 grad_clip=15.0)
