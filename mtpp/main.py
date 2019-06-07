import torch
import torch.cuda
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import numpy as np
import argparse
from utils import Logger
from model import load_model
from dataset import ATMDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--model', type=str, default='mlp', choices=['erpp', 'rmtpp', 'mlp'])
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--loss-alpha', type=float, default=0.05)
    args = parser.parse_args()

    logger = Logger('./log', args.__dict__)
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.gpu)

    dataset = ATMDataset(dir='data', device=device)
    train_iter = dataset.train_iter(batch_size=args.batch_size)
    valid_iter = dataset.train_iter(batch_size=args.batch_size)
    test_iter = dataset.test_iter(batch_size=args.batch_size)

    model = load_model(args.model, args)
    model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train(model, optimizer, scheduler, train_iter, valid_iter, args.epochs, logger, device)

    logger.log_evaluation(evaluate(model, test_iter), is_test=True)


def evaluate(model, data_iter):
    predicted_events = []
    target_events = []
    predicted_time = []
    target_time = []
    model.eval()
    for index, (input, target) in enumerate(data_iter):
        time, event = model.inference(input)
        predicted_time.extend(time.cpu().numpy().tolist())
        target_time.extend(target[0].tolist())
        predicted_events.extend(event.cpu().numpy().tolist())
        target_events.extend(target[1].tolist())
    return target_time, predicted_time, target_events, predicted_events


def train(model, optimizer, scheduler, train_iter, valid_iter, epochs, logger, device):
    for epoch in range(epochs):
        model.train()
        logger.log_new_epoch(epoch)
        scheduler.step()
        train_iter.shuffle()
        losses_list = [[], [], []]
        for index, (input, target) in enumerate(train_iter):
            model.zero_grad()
            time_loss, event_loss, merged_loss = model.loss(input, target)
            merged_loss.backward()
            optimizer.step()
            losses_list[0].append(time_loss.item())
            losses_list[1].append(event_loss.item())
            losses_list[2].append(merged_loss.item())
            if index * train_iter.batch_size * 5 // len(train_iter) != (index - 1) * train_iter.batch_size * 5 // len(train_iter):
                logger.log_train(np.mean(losses_list[0]), np.mean(losses_list[1]), np.mean(losses_list[2]))
                losses_list = [[], [], []]
        logger.log_evaluation(evaluate(model, valid_iter), is_test=False)
    logger.writer.close()


main()
