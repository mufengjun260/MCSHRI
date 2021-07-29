# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os
import time

import scipy.io as scio
import numpy as np
import torch
import torch.utils.data
import itertools
from torch.nn.parallel import DistributedDataParallel

from Dataset_cross import EMGDataset
from Network import EMGFlowNet

import logging


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.addHandler(streamHandler)
    return l


def generate_train_test_idx(total_length, train_rate):
    train_count = int(total_length * train_rate)
    train_idx = np.zeros(total_length, dtype=int)
    train_idx[:train_count] = 1
    np.random.shuffle(train_idx)
    test_index = 1 - train_idx
    return train_idx, test_index


def train_model(dataset_loader, test_dataset_loader, EMGModel, optimizer, criterion, epoch, model_path, logger, tag):
    global total_count
    best_result = 0.
    for epoch in range(epoch):
        epoch_loss = 0
        epoch_count = 0
        right = 0
        total_count = 0
        # best_acc = 0
        EMGModel.train()
        for index, data in enumerate(dataset_loader, 0):
            key = data[0].cuda()
            label = data[1].cuda()

            pred, _ = EMGModel(key)
            loss = criterion(pred, label.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            right += torch.sum(torch.argmax(pred, 1) == label.reshape(-1)).item()
            total_count += len(torch.argmax(pred, 1) == label.reshape(-1))

            epoch_loss += loss.item()
            epoch_count += 1

        acc = (right / total_count) * 100
        logger.info(
            "tag: {0}, epoch {1} count: {2}, train loss: {3} with acc: {4}".format(str(tag), epoch, total_count,
                                                                                   epoch_loss / epoch_count,
                                                                                   acc))

        EMGModel.eval()
        test_loss = 0
        test_count = 0
        right = 0
        total_count = 0

        with torch.no_grad():
            for index, data in enumerate(test_dataset_loader, 0):
                key = data[0].cuda()
                label = data[1].cuda()

                pred, _ = EMGModel(key)
                loss = criterion(pred, label.view(-1))

                right += torch.sum(torch.argmax(pred, 1) == label.reshape(-1)).item()
                total_count += len(torch.argmax(pred, 1) == label.reshape(-1))

                test_loss += loss.item()
                test_count += 1
            acc = (right / total_count) * 100

            logger.info(
                "tag: {0}, epoch {1} count: {2}, test loss: {3} with {0} test acc: {4}".format(tag, epoch, total_count,
                                                                                               test_loss / test_count,
                                                                                               acc))
        if acc > best_result:
            best_result = acc
            torch.save(EMGModel.state_dict(), model_path + "model.pth")
    logger.info(
        "tag: {0}, best test acc: {1} with {2} test samples".format(tag, best_result, total_count))
    return best_result, total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--clip_length', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--train_rate', type=int, default=0.7)
    opt = parser.parse_args()
    global_result = []
    global_count = []
    # train&test for only one-man
    current_time = time.time()
    log_path = "experiments/logs/{0}/{1}.log".format(current_time, "multi_human")
    if not os.path.exists("experiments/logs/{0}/".format(current_time)): os.makedirs(
        "experiments/logs/{0}/".format(current_time), exist_ok=True)

    logger = setup_logger('logger', log_path)
    iter_list = itertools.combinations([0, 3, 4, 5, 6], 3)
    for this_list in iter_list:
        dataset = EMGDataset(opt.clip_length, opt.step, list(this_list), with_remix=True, get_first_part=True,
                             with_extra_small=True, first_is_train=True, first_part_rate=0.999)
        test_dataset = EMGDataset(opt.clip_length, opt.step,
                                  list({0, 3, 4, 5, 6}.difference(set(this_list))),
                                  with_remix=False, get_first_part=True,
                                  with_extra_small=True, first_is_train=True, first_part_rate=0.999)

        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                                                     num_workers=0)
        test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True,
                                                          num_workers=0)

        EMGModel = EMGFlowNet(opt.clip_length).cuda()
        EMGModel = torch.nn.DataParallel(EMGModel)  # device_ids will include all GPU devices by default

        optimizer = torch.optim.Adam(EMGModel.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()
        model_path = "experiments/models/{0}/{1}/".format(current_time, "multi_human" + str(this_list))
        if not os.path.exists(model_path): os.makedirs(model_path, exist_ok=True)

        result, count = train_model(dataset_loader, test_dataset_loader, EMGModel, optimizer, criterion, 1000,
                                    model_path, logger,
                                    this_list)
        global_result.append(result)
        global_count.append(count)

    for i in range(len(global_result)):
        logger.info("{0},{1}".format(global_result[i], global_count[i]))


if __name__ == '__main__':
    main()
