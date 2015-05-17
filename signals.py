import numpy as np
import itertools
from scipy import interpolate, signal

def window(seq, n=2, skip=1):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    # Implimentation from: http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python

    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    i = 0
    for elem in it:
        result = result[1:] + (elem,)
        i = (i + 1) % skip
        if i == 0:
            yield result


def interpolate_points(points, fps=30.0, sample_freq = 250.0, axis=0):
    """ Given an matrix of waveforms, interpolate them along an axis
    such that the number of new is multiplied by (1/ratio) """
    # Define the old time space, i.e. the index of each point
    N = points.shape[axis]
    indices = np.arange(0, N)
    # Make an 'interpolation function' using scikit's interp1d
    f = interpolate.interp1d(indices, points, kind='cubic', axis=axis)
    # Define the new time axis,
    xnew = np.arange(0, N - 1, fps/sample_freq)
    return f(xnew)


def filter_unstable_movements(points):
    """ Filter unstable movements, e.g. coughing """
    """ In the paper, they removed points which had maximum movements
    greater than the "mode" of rounded maximum movements. Or something.
    This didn't really work or make sense when we tried it, so we tried
    something else, then gave up... """
    maximums = np.max(np.diff(points.T), axis=1)
    median = np.median(maximums)
    return points[:, maximums > median]


def make_filter(order=5, low_freq=0.75, high_freq=5, sample_freq=250.0):
    """ Make the butterworth filter function required by the pulse paper"""
    nyq = 0.5 * sample_freq
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    func = lambda x: signal.lfilter(b, a, x)
    func.b = b
    func.a = a
    return func



def find_periodicities(X, sample_freq=250.0):
    """ Find the periodicity of each signal in a matrix(along axis 0),
    and the associated frequencies of the periods"""

    # We're not sure if this is quite correct, but it's what the paper
    # seemed to imply...
    # This could also be made much neater, and optimised.
    X = X - np.mean(X, axis=0)
    sigs = []
    for row in X.T:
        sigs.append(row / np.std(row))
        
    X = np.array(sigs).T


    # Find the power spectrum of the signal (absolute fft squared)
    power = np.abs(np.fft.rfft(X, axis=0))**2

    cutoff = 1
    power = power[cutoff:, :]

    # Build a list of the actual frequencies corresponding to each fft index, using numpy's rfftfreq
    # n.b. This is where I'm having some trouble. I don't think I'm actually getting the right
    # numbers out for the frequencies of these signals...

    real_frequencies = np.fft.rfftfreq(
        X.shape[0],  d=(1 / (sample_freq)))[cutoff:]

    # Find the most powerful non-zero frequency in each signal
    start = 0
    max_indices = np.argmax(power[start:, :], axis=0) + start

    # The first haromic component of f = f*2
    harmonic_indices = max_indices * 2

    # Initialise arrays for return values
    periodicities = []
    frequencies = []
    i = 0

    # Loop over each signal
    for fundamental, harmonic in zip(max_indices, harmonic_indices):
        # Get the real frequency highest power component
        frequencies.append(real_frequencies[fundamental])

        # Get the total power of the highest power component and its
        # first harmonic, as a percentage of the total signal power
        period_power = np.sum(power[[fundamental, harmonic], i])
        total_power = np.sum(power[:, i])
        percentage = period_power / total_power

        # That number is (apparently) the signal periodicity
        periodicities.append(percentage)
        i += 1

    return np.array(frequencies), np.array(periodicities)


def getpeaks(x, winsize=151):
    """ Return the indices of all points in a signal which are the largest poitns in
    a window centered on themselves """
    for i, win in enumerate(window(x, winsize, 1)):
        ind = int(winsize / 2)
        if np.argmax(win) == ind:
            yield i + ind
