import numpy as np 

class LinearPredictor:
    def __init__(self, order):
        self._order = order
        self._coef  = np.zeros((order, ))
        shape = (order+1, )
        self._buf  = np.zeros(shape)
        self._cov  = np.zeros(shape)

    def reset(self):
        self._coef[:] = 0.0
        self._buf[:] = 0.0
        self._cov[:] = 0.0

    def _batchUpdateCov(self, x):
        N = x.shape[0]
        for p in range(self._cov.shape[0]):
            den = 0
            for j in range(N-p):
                den += 1
                self._cov[p] += x[j]*x[j+p]
            self._cov[p] /= den

    def _updateCov(self, x):
        self._buf    = np.roll(self._buf, 1)
        self._buf[0] = x
        alpha = 0.9999
        for p in range(self._cov.shape[0]):
            rss = self._buf[0]*self._buf[0+p]
            self._cov[p] = self._cov[p]*alpha + (1.0-alpha)*rss

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

    def update(self, smpl):
        self._updateCov(smpl)
        return self._updateCoef()

    def process(self, stream):
        '''
        Arg:
            - stream <np.ndarray> in reverse order, e.g. [xp, xp-1, ... x2, x1]
        Return:
            - predict : <float>
        '''
        predict = 0.0
        predict = -(self._coef * stream).sum()
        return predict


if __name__ == '__main__':
    import pdb
    import matplotlib.pyplot as plt

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
        lpc = LinearPredictor(order)
        lpc._batchUpdateCov(istream)
        lpc._updateCoef()

        buf = np.zeros((order, ))
        for n in range(order):
            buf = np.roll(buf, 1)
            buf[0] = istream[n]
        for n in range(order, num_smpl):
            ostream[n] = lpc.process(buf)
            estream[n] = ostream[n] - istream[n]
            buf = np.roll(buf, 1)
            buf[0] = istream[n]
        
        avg_err = (estream ** 2).mean()
        print(f'Avg Error = {avg_err}')
        print(f'Coef = {lpc._coef}')

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
        ostream = np.zeros(istream.shape)
        estream = np.zeros(istream.shape)

        order = 4
        lpc = LinearPredictor(order)

        buf = np.zeros((order, ))
        for n in range(order):
            lpc.update(istream[n])
            buf = np.roll(buf, 1)
            buf[0] = istream[n]
        for n in range(order, num_smpl):
            lpc.update(istream[n])
            ostream[n] = lpc.process(buf)
            estream[n] = ostream[n] - istream[n]
            buf = np.roll(buf, 1)
            buf[0] = istream[n]
        
        avg_err = (estream ** 2).mean()
        print(f'Avg Error = {avg_err}')
        print(f'Coef = {lpc._coef}')

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