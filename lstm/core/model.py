import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis

from lstm.core.utils import Timer
import torch
from torch import nn
import torch.optim as optim


class MyModel(nn.Module):
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        super(MyModel,self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=100, num_layers=3,dropout=0.2,batch_first=True)
        # self.lstm_2 = nn.LSTM(input_size=4, hidden_size=4)
        # self.lstm_3 = nn.LSTM(input_size=4, hidden_size=4, dropout=0.2)
        self.linear = nn.Linear(100,1)



    def forward(self,input):

        output, (hn, cn) = self.lstm(input)
        output = self.linear(output[:,-1,:])
        return output

class Model(nn.Module):
    """A class for an building and inferencing an lstm model"""


    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = torch.load(filepath)





    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted
