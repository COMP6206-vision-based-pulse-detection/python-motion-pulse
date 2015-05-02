#!/usr/local/bin/python
# coding: utf-8

import cv2
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy import interpolate, signal
import sklearn.decomposition

import threading
import Queue


class VideoStreamProducer:
    def __init__(self, maxFrameNumber, sourceType, fileName=None):
        self.frameQueue = Queue.Queue(maxFrameNumber)
        self.maxFrameNumber = maxFrameNumber
        if sourceType=="Webcam":
            self.sourceName=0
        elif sourceType=="File":
            if fileName is None:
                raise Exception("Filename not specified")
            else:
                self.sourceName=fileName
        else:
            raise Exception("Source type not specified ('Webcam' or 'File')")
    @property
    def FrameQueue(self):
        return self.frameQueue
    def ProduceFrames(self):
        """This function blocks until maxFrameNumber frames are produced"""
        camera = cv2.VideoCapture(self.sourceName)
        for _ in range(self.maxFrameNumber):
            success, frame = camera.read()
            if not success:
                print "Video stream ended unexpectedly"
                break
            else:
                self.frameQueue.put((frame, None))
        self.frameQueue.join()

class FrameProcessor(threading.Thread):
    def __init__(self,frameQueue,incrementalPCA,n_components,windowSize):
        threading.Thread.__init__(self)
        self.daemon = True
        self.frameQueue = frameQueue
        self.windowSize = windowSize
        if incremental:
            self.pca = sklearn.decomposition.IncrementalPCA(n_components=n_components)
        else:
            self.pca = sklearn.decomposition.PCA(n_components=n_components)
    def Run(self):
        self.ProcessFrames()
    def ProcessFrames(self):
        #create a butterworth filter
        butter_filter = make_filter()
        #initialise a variable for the signal to go in
        signal = np.ndarray((0, 5))
        while True:
            pass

    @staticmethod
    def get_moving_points(source_queue, do_draw=True, n_points=100):
        """ Open up a video source, find a face and track points on it.
        Every frame, yield the position delta for every point being tracked """

        # Define parameters for ShiTomasi corner detection (we have no iea if
        # these are good)
        feature_params = dict(maxCorners=n_points,
                              qualityLevel=0.1,
                              minDistance=2,
                              blockSize=7)

        # Define parameters for Lucas-Kanade optical flow (we have no iea if these
        # are good either)
        critera = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=critera)

        # Initialise a face detector using a premade XML file
        face_cascade = cv2.CascadeClassifier('faces.xml')

        # Capture the first frame, convert it to B&W
        frame = source_queue.get(block=True)[0]
        old_img = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
        #say that we've finished processing the first frame
        source_queue.task_done()

        # Build a mask which covers a detected face, except for the eys
        mask = np.zeros_like(old_img)
        faces = face_cascade.detectMultiScale(old_img, 1.3, 5)
        rects = make_face_rects(faces[0])
        for x, y, w, h in rects:
            # Fill in a rectangle area of the 'mask' array white
            cv2.rectangle(mask, (x, y), ((x + w), (y + h)),
                          thickness=-1,
                          color=(255, 255, 255))

        # Use a corner detector to find "good features to track" inside the mask
        # n.b. we're not sure if this is the right way of picking points to track
        p0 = cv2.goodFeaturesToTrack(old_img, mask=mask, **feature_params)
        firstp = p0
        # An array of random colours, for drawing things!
        color = np.random.randint(0, 255, (100, 3))

        while True:
            # Load next frame, convert to greyscale
            frame,framePeriod = source_queue.get(block=True)
            new_img = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)

            # Use the Lucas-Kande optical flow thingy to detect the optical flow
            # between the old
            p1, status, err = cv2.calcOpticalFlowPyrLK(
                old_img, new_img, p0, None, **lk_params)

            # Select points for which the flow was successfully found
            good_new = p1[status == 1]
            good_old = p0[status == 1]
            good_first = firstp[status == 1]

            # Debugging code, draw the 'flow' if do_draw == True
            if do_draw:
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.line(old_img, (a, b), (c, d), color[i].tolist(), 2)
                    cv2.circle(old_img, (a, b), 5, color[i].tolist(), -1)
                cv2.imshow("Video", old_img)
                cv2.waitKey(1)

            # Yield the y-component of the point positions
            yield (good_new - good_first)[:, 1]

            #say that we've finished with the frame
            source_queue.task_done()

            # Set the 'previous' image to be the current one
            # and the previous point positions to be the current ones
            old_img = new_img.copy()
            p0 = good_new.reshape(-1, 1, 2)

########################################################################################
    # Track some points in a video, changing over time
    for points in window(get_moving_points("face2-2.mp4", do_draw=False, n_points=50), N, N - 1):

        # Interpolate the points to 250 Hz
        interpolated = interpolate_points(np.vstack(points)).T

        # Filter unstable movements
        # interpolated = filter_unstable_movements(interpolated.T).T

        # Filter with a butterworth filter
        filtered = butter_filter(interpolated).T

        # For fitting PCA, remove the time-frames with the top 25% percentile
        # largfest mvoements
        norms = np.linalg.norm(filtered, 2, axis=1)
        removed_abnormalities = filtered[norms > np.percentile(norms, 75)]

        # Perform PCA, getting the largest 5 components of movement
        if incremental:
            pca.partial_fit(removed_abnormalities)
        else:
            pca.fit(removed_abnormalities)

        # Project the tracked point movements on to the principle component vectors,
        # producing five waveforms for the different components of movement
        transformed = pca.transform(filtered)

        signal = np.vstack((signal, transformed))

        # Find the periodicity of each signal
        frequencies, periodicities = find_periodicities(signal)

        # Find the indices of peaks in the signal
        peaks = [list(getpeaks(signal.T[i])) for i in range(5)]
        most_periodic = np.argmax(periodicities)

        # The frequency of the most periodic signal is supposedly the heart
        # rate
        print "Periodicities: ", periodicities
        print "Most periodic: ", most_periodic
        print "Frequencies: ", frequencies
        print "Peak count BPMs: ", [len(p) for p in peaks]
        print "Heart rate by FFT estimate: {} BPM".format(60.0 / frequencies[most_periodic])
        countbpm = 10.0 / \
            ((signal.shape[0] / (250.0 / 30.0)) / 30) * \
            6 * len(peaks[most_periodic])
        print signal.shape
        print "Heart rate by peak estimate: {} BPM".format(countbpm)
########################################################################################

def make_face_rects(rect):
    """ Given a rectangle (covering a face), return two rectangles.
        which cover the forehead and the area under the eyes """
    x, y, w, h = rect

    rect1_x = x + w / 4.0
    rect1_w = w / 2.0
    rect1_y = y + 0.05 * h
    rect1_h = h * 0.9 * 0.2

    rect2_x = rect1_x
    rect2_w = rect1_w

    rect2_y = y + 0.05 * h + (h * 0.9 * 0.55)
    rect2_h = h * 0.9 * 0.45

    return (
        (int(rect1_x), int(rect1_y), int(rect1_w), int(rect1_h)),
        (int(rect2_x), int(rect2_y), int(rect2_w), int(rect2_h))
    )




def window(seq, n=2, skip=1):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    # http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python

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


def interpolate_points(points, ratio=30.0 / 250.0, axis=0):
    """ Given an matrix of waveforms, interpolate them along an axis
    such that the number of new is multiplied by (1/ratio) """
    # Define the old time space, i.e. the index of each point
    N = points.shape[axis]
    indices = np.arange(0, N)
    # Make an 'interpolation function' using scikit's interp1d
    f = interpolate.interp1d(indices, points, kind='cubic', axis=axis)
    # Define the new time axis,
    xnew = np.arange(0, N - 1, ratio)
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
    return lambda x: signal.lfilter(b, a, x)


def find_periodicities(X, sample_freq=250.0):
    """ Find the periodicity of each signal in a matrix(along axis 0),
    and the associated frequencies of the periods"""

    # We're not sure if this is quite correct, but it's what the paper
    # seemed to imply...
    # This could also be made much neater, and optimised.

    # Find the power spectrum of the signal (absolute fft squared)
    power = np.abs(np.fft.rfft(X, axis=0))**2

    # Build a list of the actual frequencies corresponding to each fft index, using numpy's rfftfreq
    # n.b. This is where I'm having some trouble. I don't think I'm actually getting the right
    # numbers out for the frequencies of these signals...

    real_frequencies = np.fft.rfftfreq(
        power.shape[0],  d=(1 / (sample_freq)))

    # Find the most powerful non-zero frequency in each signal
    max_indices = np.argmax(power[1:, :], axis=0) + 1

    # The first haromic component of f = f*2
    harmonic_indices = max_indices * 2

    # Initialise arrays for return values
    periodicities = []
    frequencies = []
    i = 0

    # Loop over each signal
    for i1, i2 in zip(max_indices, harmonic_indices):
        # Get the real frequency highest power component
        frequencies.append(real_frequencies[i1])

        # Get the total power of the highest power component and its
        # first harmonic, as a percentage of the total signal power
        period_power = np.sum(power[[i1, i2], i])
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


def main(incrementalPCA=False):
    """ Run the full algorithm on a video """

    # Work on the first N frames
    if incrementalPCA:
        N = 60
        #plot = False
    else:
        N = 299
        #plot = True


    #create a video stream producer
    vstream = VideoStreamProducer(300,"File","face2-2.mp4")

    #create a video stream processor
    frameproc = FrameProcessor(vstream.FrameQueue,incrementalPCA,5,N)
    #set it going in its own thread (it will wait until it has something to process)
    frameproc.start()

    #Set the frame producer going (this will block until the frame processor has finished)
    vstream.ProduceFrames()

    """This will need fixing to work with the new threaded structure"""
    #if plot:
    #    ax = plt.subplot(3, 1, 1)
    #    ax.set_title("Interpolated point y-positions")
    #    plt.plot(interpolated.T)
    #    ax = plt.subplot(3, 1, 2)
    #    ax.set_title("Filtered point y-positions")
    #    plt.plot(filtered)
    #    ax = plt.subplot(3, 1, 3)
    #    for i, p in enumerate(periodicities):
    #        w = 5 if i == most_periodic else 1
    #        plt.plot(transformed[:, i], linewidth=w)
    #        plt.plot(peaks[i], [transformed.T[i][k] for k in peaks[i]], 'o')

    #    ax.set_title("Components of motion")



if __name__ == '__main__':
    main(False)

plt.show()

