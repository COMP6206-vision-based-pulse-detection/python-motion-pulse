import unittest
import numpy as np
import signals
import matplotlib.pyplot as plt
import math


class TestSignalFunctions(unittest.TestCase):

    def make_test_signal(self, sample_rate):
        """ Make a 10s signal with three frequency components
        f = [100, 10, 1]/(2pi) """
        x = np.arange(0, 10, 1.0 / float(sample_rate))
        y = np.sin(x * 10.0)
        y += np.sin(x * 1.0) * 0.7
        y += np.sin(x * 100.0) * 0.1
        return (x, y)

    def test_get_peaks(self):
        """ Check that get_peaks finds approximately the right locations and
        number of peaks in a sine wave """
        x = np.arange(0, math.pi * 8, 0.01)
        y = np.sin(x)
        test_peaks = x[list(signals.getpeaks(y, 151))] / (math.pi / 2.0)
        real_peaks = [1, 5, 9, 13]
        np.testing.assert_allclose(test_peaks, real_peaks, rtol=1e-3)

    def test_make_filter(self):
        """ Check that the bandpass filter function filters the correct signals.
        N.B graph must be checked by eye! """
        filt = signals.make_filter(
            order=5, low_freq=0.75, high_freq=5, sample_freq=250.0)
        x, y = self.make_test_signal(250.0)

        yf = filt(y)
        f = plt.figure()
        plt.plot(x, y)
        plt.plot(x, yf)
        plt.xlabel('Time (250 Hz sample rate)')
        plt.ylabel('Signal')
        plt.legend(['Original', 'Filtered'])
        f.savefig('test_make_filter.png', bbox_inches='tight')

    def test_window(self):
        """ Test the moving window function """
        windows = list(signals.window(xrange(10), 4, 2))
        expected = [(0, 1, 2, 3), (2, 3, 4, 5), (4, 5, 6, 7), (6, 7, 8, 9)]
        self.assertEquals(windows, expected)

    def test_interpolate_points(self):
        """ Test point interpolation
        N.B. Graph must be checked by eye! """
        x, y = self.make_test_signal(30.0)
        y = np.vstack((y, -y))
        y0 = signals.interpolate_points(y.T, 30.0 / 250.0, axis=0)
        y1 = signals.interpolate_points(y, 30.0 / 250.0, axis=1)
        self.assertTrue(y0.shape[0] >= 2490)
        self.assertTrue(y1.shape[1] >= 2490)
        f = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(y.T)
        plt.subplot(2, 1, 2)
        plt.plot(y0)
        f.savefig('test_interpolate_points.png', bbox_inches='tight')

    def test_find_periodicities(self):
        """ Test that find_periodicities picks the correct signal as the
        most periodic one, and returns a sensible frequency """
        sample_freq = 30.0
        x, y_periodic = self.make_test_signal(sample_freq)
        y_flat = x / 8.0
        y = np.array([y_periodic, y_flat])
        frequencies, periodicies = signals.find_periodicities(
            y.T, sample_freq=sample_freq)
        self.assertGreater(periodicies[0], periodicies[1])
        self.assertAlmostEqual(frequencies[0], 10 / (2 * math.pi), places=1)


if __name__ == '__main__':
    unittest.main()
