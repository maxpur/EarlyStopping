import numpy as np
import torch

# Python class to evaluate your training result of a neural network and to avoid overfitting
class EarlyStopping:
    # Initialize data
    def __init__(self, lookback, deg, debug=False):
        self.states = list()
        self.train_losses = list()
        self.val_losses = list()
        self.lookback = lookback
        self.deg = deg
        self.final_model = None
        self.debug = debug

    # Check if overfitting is happening and safe best model state
    # Return True if overfitting started
    def early_stop(self, val_loss, train_loss, model_state):
        # Log current losses 
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        self.log(f'Training loss: {train_loss}, validation loss: {val_loss}')

        if len(self.train_losses) < 2: 
            self.log("Skip early stop due to length of input array")
            self.states.append(None)
            return False

        # Define array that will be used to calculate the increasing/decreasing tendency of the data
        train_loss_window = self.train_losses[len(self.train_losses)-min(self.lookback, len(self.train_losses)):]
        val_loss_window = self.val_losses[len(self.val_losses)-min(self.lookback, len(self.val_losses)):]

        # Check if overfitting starts
        if self.is_decreasing(train_loss_window) and self.is_increasing(val_loss_window):
            # Overfitting starts
            self.log("Overfitting detected")
            min_id = self.val_losses.index(min(self.val_losses[1:]))
            self.final_model = self.states[min_id]
            return True
        else:
            # Model performance is still increasing
            self.states.append(model_state)
            return False

    # Calculate slope of interpolated data with a range defined in lookback
    # Return slope
    def get_slope(self, data):
        indices = list(range(1, len(data) + 1))
        return np.polyfit(indices, data, self.deg)[-2]

    # Check if slope is increasing
    # Return True if slope is increasing
    def is_increasing(self, data):
        return self.get_slope(data) > 0

    # Check if slope is decreasing
    # Return True if slope is decreasing
    def is_decreasing(self, data):
        return self.get_slope(data) < 0
    
    # Save best model state 
    def save_state(self, path):
        self.log("Save state")
        torch.save(self.final_model, path)

    # Get minimum training loss
    def get_min_train_loss(self):
        return min(self.train_losses)
    
    # Get minimum validation loss
    def get_min_val_loss(self):
        return min(self.val_losses)
    
    # Print debug information
    def log(self, log):
        if self.debug:
            print(f'[DEBUG] : \tEpoche {len(self.train_losses)}\t {log}')

