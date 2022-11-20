import numpy as np 
from LinearPredictor.LinearPredictorInterface import ILinearPredictor

class Levinson(ILinearPredictor):
    def __init__(self, order):
        super().__init__()
        self._order = order
        self._coef  = np.zeros((order, ))
        self._cov   = np.zeros((order+1, ))

    def _updateCov(self, x):
        self._cov[0] = (x * x).sum()
        for p in range(1, self._cov.shape[0]):
            self._cov[p] = (x[p:] * x[:-p]).sum()

    def _updateCoef(self):
        Ak  = np.zeros((self._order+1, ))
        Ak[0] = 1.0
        err = self._cov[0]
        eps = 1e-9
        for p in range(self._order):
            k = 0.0; 
            for j in range(p+1):
                k -= Ak[j]*self._cov[p+1-j]
            k /= (err + eps)

            for n in range((p+1)//2 + 1):
                temp = Ak[p+1-n] + k*Ak[n]
                Ak[n] += k*Ak[p+1-n]
                Ak[p+1-n] = temp
            err *= (1.0 - k*k)
        self._coef[:] = Ak[1:]
        return err

    def update(self, istream):
        self._updateCov(istream)
        return self._updateCoef()

    def pem(self, istream, prevStream=None):
        '''
        Predict Error Method
        -----------------------
        Arg:
            - istream <np.ndarray> : current stream
        Return:
            - error   <np.ndarray> : predicted error stream
            - buf     <np.ndarray> : buffer for next prediction
        '''
        N = istream.shape[0]
        buf = prevStream if (prevStream is not None) else np.zeros((self._order, )) 
        estream = np.zeros(istream.shape)
        for n in range(N):
            pred = self.predictNext(buf)
            estream[n] = (istream[n] - pred)
            buf = np.roll(buf, 1)
            buf[0] = istream[n]
        return estream, np.copy(buf)

    def predictNext(self, frm):
        '''
        Predict next sample based on previous observation (auto-regression)
        -----------------------
        Arg:
            - frm     : <np.ndarray> in reverse order (from new to old), e.g. [xp, xp-1, ... x2, x1]
        Return:
            - predict : <float>
        '''
        return -(self._coef * frm).sum()
    
    def getCoef(self):
        return np.copy(self._coef)


if __name__ == '__main__':
    import pdb
    import matplotlib.pyplot as plt
    from scipy import signal

    def genTestSignal(num_smpl):
        i = np.array(list(range(0, num_smpl)))
        sig = np.sin(i*0.01) + 0.75*np.sin(i*0.03) \
                + 0.5*np.sin(i*0.05) + 0.25*np.sin(i*0.11)
        return sig

    def testBatchUpdate():
        print('Test Batch Update LPC!')
        num_smpl = 128
        istream = genTestSignal(num_smpl)
        ostream = np.zeros(istream.shape)
        estream = np.zeros(istream.shape)

        order = 4
        lpc = Levinson(order)
        lpc.update(istream)

        buf = np.zeros((order, ))
        for n in range(order):
            buf = np.roll(buf, 1)
            buf[0] = istream[n]
        for n in range(order, num_smpl):
            ostream[n] = lpc.predictNext(buf)
            estream[n] = ostream[n] - istream[n]
            buf = np.roll(buf, 1)
            buf[0] = istream[n]
        
        avg_err = (estream ** 2).mean()
        print(f'Avg Error = {avg_err}')
        print(f'Coef = {lpc.getCoef()}')

        plt.figure()
        plt.plot(istream, label='Truth')
        plt.plot(ostream, label='Predict')
        plt.plot(estream, label='Error')
        plt.xlabel('Time (smpl)')
        plt.ylabel('Amplitude (smpl)')
        plt.title('Test Batch Update')
        plt.legend()

    def testStreamUpdate():
        print('Test Stream Update LPC!')
        num_smpl = 128
        istream = genTestSignal(num_smpl)

        order = 4
        lpc = Levinson(order)

        win_size = 32
        hop_size = win_size // 2
        win = signal.windows.hann(win_size+1)[:-1]
        buf = np.zeros((win_size, ))
        buf_pred = np.zeros((order, ))
        ostream = np.zeros(istream.shape)
        estream = np.zeros((num_smpl+hop_size, ))
        num_frm = num_smpl // hop_size
        for n in range(num_frm):
            src = n * hop_size
            dst = (n+1)*hop_size
            buf[:hop_size] = buf[hop_size:]
            buf[hop_size:] = istream[src:dst]

            lpc.update(buf * win)
            errFrm, buf_pred = lpc.pem(buf*win, buf_pred)
            estream[src:src+win_size] += errFrm
        estream = estream[hop_size:]    # get rid of latency
        ostream = istream - estream
        avg_err = (estream ** 2).mean()
        print(f'Avg Error = {avg_err}')
        print(f'Coef = {lpc.getCoef()}')

        plt.figure()
        plt.plot(istream, label='Truth')
        plt.plot(ostream, label='Predict')
        plt.plot(estream, label='Error')
        plt.xlabel('Time (smpl)')
        plt.ylabel('Amplitude (smpl)')
        plt.title('Test Stream Update')
        plt.legend()

    testBatchUpdate()
    testStreamUpdate()
    plt.show()