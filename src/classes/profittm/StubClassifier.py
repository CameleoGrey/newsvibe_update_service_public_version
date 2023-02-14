
import numpy as np

class StubClassifier():
    
    def __init__(self, stub_y):
        
        self.stub_y = stub_y
    
    def predict(self, x):
        
        stub_predicts = [ self.stub_y for i in range(len(x)) ]
        stub_predicts = np.array( stub_predicts )
        
        return stub_predicts
    
    def fit(self, x, y):
        
        return self