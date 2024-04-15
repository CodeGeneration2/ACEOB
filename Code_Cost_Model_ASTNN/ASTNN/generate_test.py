import json
import pandas as pd
import torch
import numpy as np
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

from model import BatchProgramClassifier

from config import *
import os

# ######################################################################################################################
best_model_path = ""
data_root_path = 'ACEOB-NPI-data-Pre/'

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

    print('test..', flush=True)

    # ===================================================================================================
    def calculate_loss_accuracy(dataset, set_name):
        error_dict = {'predictions': [], 'labels': []}

        # Set the model to evaluation mode
        model.eval()

        # Skip gradient calculation with torch.no_grad()
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
                batch = get_batch(dataset, i, BATCH_SIZE)
                inputs, labels = batch
                if USE_GPU:
                    inputs, labels = inputs.cuda(), labels.cuda()

                model.batch_size = len(labels)
                model.hidden = model.init_hidden()
                model_output = model(inputs)

                model_output = model_output.squeeze()

                # Convert tensor to list
                model_output_list = model_output.tolist()
                labels_list = labels.tolist()

                error_dict['predictions'].extend(model_output_list)
                error_dict['labels'].extend(labels_list)

        return error_dict

    # ---------------------------------
    model.load_state_dict(torch.load(best_model_path))

    error_dict = calculate_loss_accuracy(test_data, "test")

    # Save dictionary to file
    with open('dict.json', 'w') as f:
        json.dump(error_dict, f)

# ######################################################################################################################
if __name__ == '__main__':
    run()
