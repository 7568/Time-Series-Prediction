__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import MyModel
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from numpy import newaxis
import pandas as pd
import  sklearn as sk
from tqdm import tqdm


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def predict_point_by_point(good_model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    print('[Model] Predicting Point-by-Point...')
    predicted = good_model(torch.tensor(data,dtype=torch.float32)).detach().numpy()
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequences_multiple(good_model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    print('[Model] Predicting Sequences Multiple...')
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            out_put = good_model(torch.tensor(curr_frame[newaxis, :, :], dtype=torch.float32)).detach().numpy()
            predicted.append(out_put[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def predict_sequence_full(good_model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    print('[Model] Predicting Sequences Full...')
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        out_put = good_model(torch.tensor(curr_frame[newaxis, :, :], dtype=torch.float32)).detach().numpy()
        predicted.append(out_put[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
    return predicted


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    '''
	# in-memory training
	model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
	'''


    rnn = MyModel()
    # rnn = torch.load('data/8_new.pt')
    criterion = nn.MSELoss()
    optimizer = optim.SGD(rnn.parameters(), lr=0.1)
    for epoch in range(10):
        print(f'epoch : {epoch}')
        loss_all = []
        train_batch = data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        )
        for x_batch, y_batch in tqdm(iterable=train_batch,total=int((data.len_train - configs['data']['sequence_length'])/2)):
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            y_batch = torch.tensor(y_batch, dtype=torch.float32)
            # print(x_batch.shape)
            out_put = rnn(x_batch)
            loss = criterion(out_put, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all.append(loss.item())
            # count += 1
            # if count == 20:
            #     print(np.mean(loss_all))
            #     count = 0
            #     loss_all = []
        print(np.mean(loss_all))
        torch.save(rnn, f'data/{epoch}_new.pt')
    # model.train_generator(
    #     data_gen=data.generate_train_batch(
    #         seq_len=configs['data']['sequence_length'],
    #         batch_size=configs['training']['batch_size'],
    #         normalise=configs['data']['normalise']
    #     ),
    #     epochs=configs['training']['epochs'],
    #     batch_size=configs['training']['batch_size'],
    #     steps_per_epoch=steps_per_epoch,
    #     save_dir=configs['model']['save_dir']
    # )



def test():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    rnn = torch.load('data/9_new.pt')
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    predictions3 = predict_point_by_point(rnn, x_test)
    from sklearn.metrics import mean_squared_error
    score = mean_squared_error(np.array(y_test.reshape(655)), np.array(predictions3))
    # predictions3 = predictions3.reshape((655, 1))
    print(score)
    predictions1 = predict_sequences_multiple(rnn, x_test, configs['data']['sequence_length'],
                                              configs['data']['sequence_length'])
    predictions2 = predict_sequence_full(rnn, x_test, configs['data']['sequence_length'])

    plot_results_multiple(predictions1, y_test, configs['data']['sequence_length'])
    plot_results(predictions2, y_test)
    plot_results(predictions3, y_test)


if __name__ == '__main__':
    # main()
    test()
