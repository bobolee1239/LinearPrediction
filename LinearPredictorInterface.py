import numpy as np 

class ILinearPredictor:
    def __init__(self):
        pass

    def update(self, istream):
        err = np.copy(istream)
        return err

    def pem(self, istream):
        '''
        Arg:
            - stream <np.ndarray> in reverse order, e.g. [xp, xp-1, ... x2, x1]
        Return:
            - error : <np.ndarray>
        '''
        predict_error = np.copy(istream)
        return predict_error

    def predictNext(self, frm):
        return 0
    
    def getCoef(self):
        return None