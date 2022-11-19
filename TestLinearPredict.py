import os
import numpy as np 
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt

from scipy import signal
from Levinson import Levinson

def genExcitation(size, period):
    excitation = np.zeros((size, ))
    cnt = 0
    while cnt < size:
        excitation[cnt] = 0.028
        cnt += period
    return excitation

def genHannWin(size):
    return signal.windows.hann(size+1)[:-1]

class SimulateParam:
    def __init__(self, lpc, win_size):
        self._lpc = lpc
        self._win = genHannWin(win_size)
    
    def getWindow(self):
        return self._win

    def getLinearPredictor(self):
        return self._lpc


def testLpAnalyze(istream, sim_param):
    print('[Test Linear Predictor Analyze]')
    num_smpl = istream.shape[0]

    lpc = sim_param.getLinearPredictor()
    lpc_order = lpc.getCoef().shape[0]

    win = sim_param.getWindow()
    win_size = win.shape[0]
    hop_size = win_size // 2
    frm = np.zeros((win_size, ))
    buf = np.zeros((lpc_order, ))

    ostream = np.zeros((num_smpl, ))
    estream = np.zeros((num_smpl+hop_size, ))
    num_frm = num_smpl // hop_size
    for n in range(num_frm):
        src = n * hop_size
        dst = (n+1)*hop_size
        frm[:hop_size] = frm[hop_size:]
        frm[hop_size:] = istream[src:dst]

        lpc.update(frm * win)
        errFrm, buf = lpc.pem(frm*win, buf)
        estream[src:src+win_size] += errFrm
    estream = estream[hop_size:]    # get rid of latency
    ostream = istream - estream

    plt.figure()
    plt.plot(istream, label='Truth')
    plt.plot(ostream, label='Predict')
    plt.plot(estream, label='Error')
    plt.xlabel('Time (smpl)')
    plt.ylabel('Amplitude (smpl)')
    plt.ylim([-0.9, 0.9])
    plt.title('Test Linear Predictor Analyze')
    plt.legend()

    return ostream, estream 


def testLpSynthesize(istream, excitation, sim_param):
    print('[Test Linear Predictor Synthesize]')
    num_smpl = istream.shape[0]
    ostream = np.zeros(istream.shape)

    lpc = sim_param.getLinearPredictor()
    lpc_order = lpc.getCoef().shape[0]

    win = sim_param.getWindow()
    win_size = win.shape[0]
    hop_size = win_size // 2
    frm = np.zeros((win_size, ))

    buf = np.zeros((lpc_order, ))
    num_frm = num_smpl // hop_size
    for n in range(num_frm):
        src = n * hop_size
        dst = (n+1)*hop_size
        frm[:hop_size] = frm[hop_size:]
        frm[hop_size:] = istream[src:dst]

        lpc.update(frm * win)
        for n in range(src, dst):
            ostream[n] = lpc.predictNext(buf) + excitation[n]
            buf = np.roll(buf, 1)
            buf[0] = ostream[n]
    plt.figure()
    plt.plot(istream, label='Truth')
    plt.plot(ostream, label='Synth')
    plt.xlabel('Time (smpl)')
    plt.ylabel('Amplitude (smpl)')
    plt.ylim([-0.9, 0.9])
    plt.title('Test Linear Predictor Synthesize')
    plt.legend()
    return ostream

def main(args):
    iFile = args.input 
    oFile = args.output
    oFname, _ = os.path.splitext(oFile)

    istream, sr = sf.read(iFile)
    if len(istream.shape) > 1:
        istream = istream[:, 0]

    lpc_order = 20
    win_size  = 32
    assert(win_size > lpc_order), f'[ERROR] Window size ({win_size}) must be greater than Lp order ({lpc_order})!'
    ostream, estream = testLpAnalyze(
                        istream, 
                        SimulateParam(
                            Levinson(lpc_order),
                            win_size
                        ))
    avg_err = (estream**2).mean()
    print(f'- Analyze Error = {avg_err}')
    sf.write(f'{oFname}_anlyz_voc.wav', ostream, sr)
    sf.write(f'{oFname}_anlyz_err.wav', estream, sr)

    excitation = genExcitation(istream.shape[0], period=64)
    ostream = testLpSynthesize(
                istream, 
                excitation,
                SimulateParam(
                    Levinson(lpc_order),
                    win_size
                ))
    sf.write(f'{oFname}_synth.wav', ostream, sr)
    plt.show()


if __name__ == '__main__':
    import pdb
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=True,
        help='Input wav file'
        )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help='Output wav file'
        )
    args = parser.parse_args()
    main(args)
