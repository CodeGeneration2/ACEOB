import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

from model import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import *
import os
import sys
import copy

# ######################################################################################################################
# -----------------------------------
start_epoch = 0
model_save_path = "Model_Pre_NPI"
os.makedirs(model_save_path, exist_ok=True)

# ######################################################################################################################
def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item.iloc[1])
        labels.append(item.iloc[2]-1)
    return data, torch.FloatTensor(labels)

# ######################################################################################################################
def run():
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    data_root_path = 'ACEOB-NPI-data-Pre/'
    train_data = pd.read_pickle(data_root_path + 'train/blocks.pkl')
    test_data = pd.read_pickle(data_root_path + 'test/blocks.pkl')

    word2vec = Word2Vec.load(data_root_path + "train/embedding/node_w2v_128").wv
    total_words = word2vec.vectors.shape[0]
    embedding_size = word2vec.vectors.shape[1]

    embeddings = np.zeros((total_words + 1, embedding_size), dtype="float32")
    embeddings[:total_words] = word2vec.vectors

    model = BatchProgramClassifier(embedding_size,
                                   HIDDEN_DIM,
                                   total_words + 1,
                                   ENCODE_DIM,
                                   LABELS,
                                   BATCH_SIZE,
                                   USE_GPU,
                                   embeddings,
                                   TASK
                                   )
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.MSELoss()

    training_loss_list = []

    print('train...', flush=True)
    # Training procedure

    # ===================================================================================================
    def train_return_loss_accuracy(dataset, set_name):
        total_loss = 0.0
        total = 0.0
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = get_batch(dataset, i, BATCH_SIZE)
            inputs, labels = batch
            if USE_GPU:
                inputs, labels = inputs.cuda(), labels.cuda()

            if set_name == "train":
                model.zero_grad()
            model.batch_size = len(labels)
            model.hidden = model.init_hidden()
            model_output = model(inputs)

            model_output = model_output.squeeze()
            loss = loss_function(model_output, Variable(labels.float()))

            if set_name == "train":
                loss.backward()
                optimizer.step()

            total += len(labels)
            total_loss += loss.item() * len(inputs)

        return total_loss, total

    # ===================================================================================================
    for epoch in tqdm(range(start_epoch, EPOCHS)):
        print(f'epoch: {epoch} ...')

        epoch_start_time = time.time()

        train_loss, total = train_return_loss_accuracy(train_data, "train")

        training_loss_list.append(train_loss / total)

        # ===================================================================================================
        epoch_end_time = time.time()
        print(f'[Epoch: {epoch}] Training Loss: {training_loss_list[epoch]:.2f}, Time Spent: {epoch_end_time - epoch_start_time:.2f} s', flush=True)

        torch.save(model.state_dict(), f'{model_save_path}/ASTNN_Pre_NPI_{epoch}.pth')

    # ===================================================================================================
    test_loss, total = train_return_loss_accuracy(test_data, "test")
    print("Testing results(Loss):", test_loss / total)

    torch.save(model.state_dict(), f'{model_save_path}/ASTNN_Pre_NPI.pth')

# ######################################################################################################################
if __name__ == '__main__':
    run()
