import numpy as np
import math

class PredictionHandler:
    def __init__(self, prediction, mask = None):
        self.prediction = prediction
        if mask is not None:
            self.setMask(mask) 
    
    def setMask(self, mask):
        if not mask.any():
            raise Exception("No actions")
        self.mask = mask
        return self
        
    def applyMask(self, fillValue = 0):
        if self.mask is None:
            return self.prediction
        return np.array([self.prediction[i] if self.mask[i] else fillValue for i in range(len(self.prediction))])
        
    def getBestAction(self):
        prediction = self.applyMask(-math.inf)
        return prediction.argmax()
    
    def getMaskedPrediction(self):
        prediction = self.applyMask(0)
        predictionSum = prediction.sum()
        if not predictionSum == 0:
            prediction = prediction / predictionSum
        else:
            prediction = self.mask / self.mask.sum()
        return prediction

    def getRandomAction(self):
        prediction = self.applyMask(0)
        predictionSum = prediction.sum()
        if not predictionSum == 0:
            prediction = prediction / predictionSum
        else:
            prediction = self.mask / self.mask.sum()
        return np.random.choice(len(prediction), p=prediction)
    
