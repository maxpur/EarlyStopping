import numpy as np

# Python class to evaluate your training result of a neural network and to avoid overfitting
class EarlyStopping:
    # Initialize data
    def __init__(self, lookback, deg):
        self.states = list()
        self.train_losses = list()
        self.val_losses = list()
        self.lookback = lookback
        self.deg = deg
        self.final_model = None

    # Check if overfitting is happening and safe best model state
    # Return True if overfitting started
    def early_stop(self, val_loss, train_loss, model_state):
        # Log current losses 
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Define array that will be used to calculate the increasing/decreasing tendency of the data
        train_loss_window = self.train_losses[:-min(self.lookback, len(self.train_losses))]
        val_loss_window = self.val_losses[:-min(self.lookback, len(self.val_losses))]

        # Check if overfitting starts
        if self.is_decreasing(train_loss_window) and self.is_increasing(val_loss_window):
            # Overfitting starts
            min_id = self.val_losses.index(min(self.val_losses))
            self.final_model = self.states[min_id]
            return True
        else:
            # Model performance is still increasing
            self.states.append(model_state)
            return False

    # Calculate slope of interpolated data with a range defined in lookback
    # Return slope
    def get_slope(self, data):
        indices = [range(1, len(data) + 1)]
        return np.polyfit(indices, data, self.deg)[-2]

    # Check if slope is increasing
    # Return True if slope is increasing
    def is_increasing(self, data):
        return self.get_slope(data) > 0

    # Check if slope is decreasing
    # Return True if slope is decreasing
    def is_decreasing(self, data):
        return self.get_slope(data) < 0
    

        

