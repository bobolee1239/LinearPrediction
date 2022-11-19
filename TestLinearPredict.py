
import numpy as np 
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt

from scipy import signal
from LinearPredictor import LinearPredictor

def genExcitation(size, period):
    excitation = np.zeros((size, ))
    cnt = 0
    while cnt < size:
        excitation[cnt] = 0.01
        cnt += period
    return excitation

def genHannWin(size):
    return signal.windows.hann(size+1)[:-1]

def testLpAnalyze(istream, lpc_order):
    print('[Test Linear Predictor Analyze]')
    num_smpl = istream.shape[0]
    ostream = np.zeros(istream.shape)
    estream = np.zeros(istream.shape)
    lpc = LinearPredictor(lpc_order)

    win_size = 64
    hop_size = win_size // 2
    win = genHannWin(win_size)
    frm = np.zeros((win_size, ))
    num_frm = num_smpl // hop_size

    for n in range(num_frm):
        src = n * hop_size
        dst = (n+1)*hop_size
        frm[:hop_size] = frm[hop_size:]
        frm[hop_size:] = istream[src:dst]

        lpc.update(frm * win)
        errFrm = lpc.pem(frm)
        estream[src:dst] = errFrm[:hop_size]
        ostream[src:dst] = istream[src:dst] - estream[src:dst]

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


def testLpSynthesize(istream, lpc_order, excitation):
    print('[Test Linear Predictor Synthesize]')
    num_smpl = istream.shape[0]
    ostream = np.zeros(istream.shape)

    lpc = LinearPredictor(lpc_order)
    buf = np.zeros((lpc_order, ))

    win_size = 64
    hop_size = win_size // 2
    win = genHannWin(win_size)
    frm = np.zeros((win_size, ))
    num_frm = num_smpl // hop_size

    buf = np.zeros((lpc_order, ))
    for n in range(num_frm):
        src = n * hop_size
        dst = (n+1)*hop_size
        frm[:hop_size] = frm[hop_size:]
        frm[hop_size:] = istream[src:dst]

        lpc.update(frm * win)
        for n in range(src, dst):
            ostream[n] = lpc.predict(buf) + excitation[n]
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

    istream, sr = sf.read(iFile)
    if len(istream.shape) > 1:
        istream = istream[:, 0]

    lpc_order = 20
    ostream, estream = testLpAnalyze(istream, lpc_order)
    avg_err = (estream**2).mean()
    print(f'[Test Lp Analzye] Predict Err={avg_err}')

    excitation = genExcitation(istream.shape[0], period=64)
    ostream = testLpSynthesize(istream, lpc_order, excitation)
    sf.write(oFile, ostream, sr)
    plt.show()
    pdb.set_trace()

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
