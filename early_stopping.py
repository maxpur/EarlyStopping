import torch
import numpy as np


class EarlyStopping:
    def __init__(self, lookback, tolerance, path):
        self.states = list()
        self.train_losses = list()
        self.val_losses = list()
        self.lookback = lookback
        self.tolerance = tolerance
        self.path = path

    def early_stop(self, val_loss, train_loss, model):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if self.check_sequence(self.val_losses) and self.check_sequence(self.train_losses):
            self.states.append(model)
            return True
        else:
            id = self.train_losses.index(min(self.train_losses))
            torch.save(self.states[id], self.path)   
            return False  

    def check_sequence(self, data):
        current = data[- min(self.lookback, len(self.val_losses)):]
        return self.check_tendency(current) and self.check_tolerance(current)
            
    def check_tolerance(self, data):
        tmp = data[0]
        for value in data:
            if abs(tmp - value) > self.tolerance:
                return False
            tmp = value
        return True
    
    def check_tendency(self, data):
        return np.polyfit(range(1,len(data) + 1), data, 1)[-2] < 0
    
# TODO: Only return false if training error continues to decrease and validation error start to increase
# TODO: Search minimum in the subarray [:len(array) - 1]

