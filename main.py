#!/usr/local/bin/python
# coding: utf-8

import cv2
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy import interpolate, signal
import sklearn.decomposition
from signals import *
from facetracking import *



def get_moving_points(video_source="face2-2.mp4", do_draw=True, n_points=100):
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

    # Open up the video source. 0 = webcam
    camera = cv2.VideoCapture(video_source)

    # Initialise a face detector using a premade XML file
    face_cascade = cv2.CascadeClassifier('faces.xml')

    # Capture the first frame, convert it to B&W
    go, capture = camera.read()
    old_img = cv2.cvtColor(capture, cv2.cv.CV_BGR2GRAY)

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

    go, capture = camera.read()
    while go:

        # Load next frame, convert to greyscale
        new_img = cv2.cvtColor(capture, cv2.cv.CV_BGR2GRAY)

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

        # Set the 'previous' image to be the current one
        # and the previous point positions to be the current ones
        old_img = new_img.copy()
        p0 = good_new.reshape(-1, 1, 2)
        # Continue round the loop!
        go, capture = camera.read()


def main(incremental=False):
    """ Run the full algorithm on a video """

    # I did the windowing thing thinking that we'd be able to work on small windows at once
    # and return estimtes of the pulse in real time...
    # However, with the algorithm given in the pulse paper, we can't yet, because we need to
    # process a good few seconds at once.
    # However, we could change things by using incremental PCA...

    # Work on the first N frames
    butter_filter = make_filter()
    signal = np.ndarray((0, 5))
    if incremental:
        N = 60
        plot = False
        pca = sklearn.decomposition.IncrementalPCA(n_components=5)
    else:
        N = 299
        plot = True
        pca = sklearn.decomposition.PCA(n_components=5)

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

    if plot:
        ax = plt.subplot(3, 1, 1)
        ax.set_title("Interpolated point y-positions")
        plt.plot(interpolated.T)
        ax = plt.subplot(3, 1, 2)
        ax.set_title("Filtered point y-positions")
        plt.plot(filtered)
        ax = plt.subplot(3, 1, 3)
        for i, p in enumerate(periodicities):
            w = 5 if i == most_periodic else 1
            plt.plot(transformed[:, i], linewidth=w)
            plt.plot(peaks[i], [transformed.T[i][k] for k in peaks[i]], 'o')

        ax.set_title("Components of motion")


if __name__ == '__main__':
    main(False)

plt.show()
