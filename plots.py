import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy import interpolate, signal
import sklearn.decomposition
import signals
import facetracking
import threading
import Queue
import time
import pickle

t = 6

data = pickle.load(open('all.pkl', 'r'))
fft = pickle.load(open('fft.pkl', 'r'))

print data['transformed'].T.shape
print data['filtered'].T.shape

plt.figure()
plt.plot(data['points'][0:(t * 30)])
plt.title('Facetracked points (one second, 30Hz)')
plt.xlabel('Frame number')
plt.ylabel('Position offset')
plt.savefig('graph1.png', bbox_inches='tight')

plt.figure()
plt.plot(data['interpolated'].T[0:(t * 250)])
plt.title('Interpolated points (one second, 250Hz)')
plt.xlabel('Interpolated frame number')
plt.ylabel('Position Offset')
plt.savefig('graph2.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(data['filtered'][0:(t * 250)])
plt.title('Butterworth filtered points')
plt.xlabel('Interpolated frame number')
plt.ylabel('Filtered point position')
plt.savefig('graph3.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(data['transformed'][0:(t * 250)])
plt.title('Princple Components of movements')
plt.xlabel('Interpolated frame number')
plt.ylabel('Waveform value')
plt.savefig('graph4.png', bbox_inches='tight')
plt.show()


plt.figure()
plt.plot(fft['frequencies'][:50] * 60.0, fft['power'][:50])
plt.title('Frequencies of principle components')
plt.xlabel('Frequency (BPM)')
plt.ylabel('Power')
plt.savefig('graph5.png', bbox_inches='tight')
plt.show()


plt.figure()
most_periodic = np.argmax(data['periodicities'])
for i, p in enumerate(data['transformed'].T):
    plt.plot(p[0:t * 250], linewidth=5 if i == most_periodic else 1)

plt.title('Princple Components of movements')
plt.xlabel('Interpolated frame number')
plt.ylabel('Waveform value')
plt.savefig('graph6.png', bbox_inches='tight')
plt.show()
