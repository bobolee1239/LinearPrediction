
import numpy as np 
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from LinearPredictor import LinearPredictor

def genExcitation(size, period):
    excitation = np.zeros((size, ))
    cnt = 0
    while cnt < size:
        excitation[cnt] = 0.1
        cnt += period
    return excitation

def main(args):
    iFile = args.input 
    oFile = args.output

    istream, sr = sf.read(iFile)
    if len(istream.shape) > 1:
        istream = istream[:, 0]

    order = 10
    lpc = LinearPredictor(order)
    # lpc._batchUpdateCov(istream)
    # lpc._updateCoef()
    num_smpl = istream.shape[0]

    period = 5
    ostream = np.zeros(istream.shape)
    # ostream = genExcitation(num_smpl, period)
    estream = np.zeros(istream.shape)
    buf     = np.zeros((order, ))
    for n in range(order):
        err = lpc.update(istream[n])
        buf = np.roll(buf, 1)
        buf[0] = istream[n]
    for n in range(order, num_smpl):
        if n % 480 == 0:
            print(f'smpl[{n}]')
        err = lpc.update(istream[n])
        ostream[n] = lpc.process(buf)
        estream[n] = (istream[n] - ostream[n])
        # print(ostream[n], istream[n])
        buf = np.roll(buf, 1)
        buf[0] = istream[n]
    # buf     = np.zeros((order, ))
    # estream = np.random.normal(0.0, 0.01, num_smpl)
    # for n in range(order, num_smpl):
    #     ostream[n] = lpc.process(buf) + estream[n]
    #     # print(ostream[n], istream[n])
    #     buf = np.roll(buf, 1)
    #     buf[0] = ostream[n]
        
    sf.write(oFile, ostream, sr)
    avg_err = (estream**2).mean()
    print(f'Predict Err={avg_err}')

    pdb.set_trace()

    plt.figure()
    plt.plot(istream, label='Truth')
    plt.plot(ostream, label='Predict')
    plt.plot(estream, label='Error')
    plt.xlabel('Time (smpl)')
    plt.ylabel('Amplitude (smpl)')
    plt.ylim([-0.9, 0.9])
    plt.legend()
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
