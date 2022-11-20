import numpy as np 

class ILinearPredictor:
    def __init__(self):
        pass

    def update(self, istream):
        err = np.copy(istream)
        return err

    def pem(self, istream):
        '''
        Predict Error Method
        -----------------------
        Arg:
            - istream <np.ndarray> : current stream
        Return:
            - error   <np.ndarray> : predicted error stream
        '''
        predict_error = np.copy(istream)
        return predict_error

    def predictNext(self, frm):
        '''
        Predict next sample based on previous observation (auto-regression)
        -----------------------
        Arg:
            - frm     : <np.ndarray> in reverse order (from new to old), e.g. [xp, xp-1, ... x2, x1]
        Return:
            - predict : <float>
        '''
        return 0
    
    def getCoef(self):
        return None