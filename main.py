#!/usr/local/bin/python
# coding: utf-8

import cv2
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


class VideoStreamProducer:

    def __init__(self, maxFrameNumber, sourceType, fileName=None,
                 maxFPS=None, printSteps=False):
        self.frameQueue = Queue.Queue(maxFrameNumber)
        self.pointsQueue = Queue.Queue()
        self.dataQueue = Queue.Queue()

        self.maxFrameNumber = maxFrameNumber
        self.sourceType = sourceType
        self.drawFaceTrack = True
        self.pointsToDraw = None
        self.dataToDraw = None

        if self.sourceType == "Webcam":
            self.sourceName = 0
        elif self.sourceType == "File":
            if fileName is None:
                raise Exception("Filename not specified")
            else:
                self.sourceName = fileName
        else:
            raise Exception("Source type not specified ('Webcam' or 'File')")
        self.maxFPS = maxFPS
        self.printSteps = printSteps
        if self.printSteps:
            filename = " '{}'".format(self.sourceName) if self.sourceType == "File" else ""
            print "VideoStreamProducer created. Source = {type}{name}".format(type=self.sourceType, name=filename)
        self.pointColours = np.random.randint(0, 255, (100, 3))
    @property
    def FrameQueue(self):
        return self.frameQueue

    def DrawFrame(self, frame):
        canvas = frame.copy()

        if self.pointsToDraw:
            for i, (new, old) in enumerate(self.pointsToDraw):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(canvas, (a, b), (c, d), self.pointColours[i].tolist(), 2)
                cv2.circle(canvas, (a, b), 5, self.pointColours[i].tolist(), -1)
        if self.dataToDraw is not None:

            cv2.putText(canvas, "{} BPM".format(int(self.dataToDraw['bpm'])), (50, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)


            sig = self.dataToDraw['signal'][:, self.dataToDraw['most_periodic']][-1500:]
            #sig = sig * 20 + 300
            sig = sig - np.mean(sig)
            sig = sig / np.std(sig)
            sig *= 50
            sig += 400

            signals.window(sig, 2, 1)
            for i, points in enumerate(signals.window(sig, 2, 1)):
                y0, y1 = points
                cv2.line(canvas, (int(i/3), int(y0)), (int(i/3)+1, int(y1)), (100, 100, 255), 2)
                

        cv2.imshow("Video", canvas)
        cv2.waitKey(1)
    def ProduceFrames(self):
        """This function blocks until maxFrameNumber frames are produced"""
        

        if self.printSteps:
            print "VideoStreamProducer opening video source"
        camera = cv2.VideoCapture(self.sourceName)
        # if we are reading from a webcam, we need to calculate FPS manually,
        #   and for that we need to measure time differences
        if self.sourceType == "Webcam":
            oldTime = time.time()
        # go through all the frames we need to collect

        for i in range(self.maxFrameNumber):
            # see if there is a maximum frame period - if so, wait for that
            #   long. The reason we do this before we read & put the first
            #   frame is that it means the first FPS measurement is at least
            #   aproximately right
            if (self.maxFPS is not None):
                time.sleep(1.0 / self.maxFPS)
            # read from the source
            success, frame = camera.read()
            # if there was an error, break the loop
            if not success:
                print "Video stream ended unexpectedly"
                break

            try:
                self.pointsToDraw = self.pointsQueue.get(block=False)
                self.pointsQueue.task_done()
            except Queue.Empty:
                pass
            try:
                self.dataToDraw = self.dataQueue.get(block=False)
                self.dataQueue.task_done()
            except Queue.Empty:
                pass

            if self.drawFaceTrack:
                self.DrawFrame(frame)



            # measure the fps
            if self.sourceType == "Webcam":
                currentTime = time.time()
                fps = 1.0 / (currentTime - oldTime)
                oldTime = currentTime
            elif self.sourceType == "File":
                fps = camera.get(cv2.cv.CV_CAP_PROP_FPS)
            # print if necessary
            if self.printSteps:
                print "Put frame " + str(i) + " into queue. FPS = " + str(fps)
            # put the frame into the queue
            self.frameQueue.put((frame, fps))
        if self.printSteps:
            print "VideoStreamProducer finished loading frames; getting ready to join threads"
        self.frameQueue.join()


class FrameProcessor(threading.Thread):

    def __init__(self, frameQueue, pointsQueue, dataQueue, incrementalPCA, n_components, drawFaceTrack,
                 n_trackPoints, windowSize, windowSkip, printSteps=False):
        threading.Thread.__init__(self)
        self.daemon = True
        self.frameQueue = frameQueue
        self.incrementalPCA = incrementalPCA
        self.pointsQueue = pointsQueue
        self.dataQueue = dataQueue
        if self.incrementalPCA:
            self.pca = sklearn.decomposition.IncrementalPCA(n_components=n_components)
        else:
            self.pca = sklearn.decomposition.PCA(n_components=n_components)
        self.n_components = n_components
        self.drawFaceTrack = drawFaceTrack
        self.n_trackPoints = n_trackPoints
        self.windowSize = windowSize
        self.windowSkip = windowSkip
        self.printSteps = printSteps
        self.most_periodic = 1
        self.bpm = 0
        if self.printSteps:
            print "FrameProcessor created"

    def run(self):
        self.ProcessFrames()

    def ProcessFrames(self):
        if self.printSteps:
            print "FrameProcessor running"
        movingPoints = self.GetMovingPoints()
        self.MeasureMovingPoints(movingPoints)

    def GetMovingPoints(self):
        """ Using the source, find a face and track points on it.
        Every frame, yield the position delta for every point being tracked """

        # Define parameters for ShiTomasi corner detection (we have no iea if
        # these are good)
        # feature_params = dict(maxCorners=self.n_trackPoints,
        #                       qualityLevel=0.1,
        #                       minDistance=2,
        #                       blockSize=7)

        feature_params = dict(maxCorners=self.n_trackPoints,
                              qualityLevel=0.01,
                              minDistance=0.01,
                              blockSize=15)

        # Define parameters for Lucas-Kanade optical flow (we have no iea if these
        # are good either)
        critera = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=critera)

        # Initialise a face detector using a premade XML file
        face_cascade = cv2.CascadeClassifier('faces.xml')

        # Capture the first frame, convert it to B&W
        # Build a mask which covers a detected face, except for the eys
        faces = ()
        print " *** Searching for a face... *** "
        old_img = None
        i = 0
        while len(faces) == 0:
            frame = self.frameQueue.get(block=True)[0]
            old_img = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
            if i == 0:
                faces = face_cascade.detectMultiScale(old_img, 1.3, 5)
            self.frameQueue.task_done()
            i += 1
            i = i % 30

        print " *** Found face *** "
        # Build a mask which covers a detected face, except for the eyes
        rects = facetracking.make_face_rects(faces[0])
        mask = np.zeros_like(old_img)
        for x, y, w, h in rects:
            # Fill in a rectangle area of the 'mask' array white
            cv2.rectangle(mask, (x, y), ((x + w), (y + h)),
                          thickness=-1,
                          color=(255, 255, 255))

        # Use a corner detector to find "good features to track" inside the mask
        # n.b. we're not sure if this is the right way of picking points to
        # track
        p0 = cv2.goodFeaturesToTrack(old_img, mask=mask, **feature_params)
        firstp = p0

        while True:
            # Load next frame, convert to greyscale
            frame, fps = self.frameQueue.get(block=True)
            new_img = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)

            # Use the Lucas-Kande optical flow thingy to detect the optical flow
            # between the old
            p1, status, err = cv2.calcOpticalFlowPyrLK(
                old_img, new_img, p0, None, **lk_params)

            # Select points for which the flow was successfully found
            good_new = p1[status == 1]
            good_old = p0[status == 1]
            good_first = firstp[status == 1]

            # Debugging code, draw the 'flow' if necessary
            self.pointsQueue.put(zip(good_new, good_old))

            # Yield the y-component of the point positions, and the fps
            yield ((good_new - good_first)[:, 1], fps)

            # say that we've finished with the frame
            self.frameQueue.task_done()

            self.dataQueue.put({'signal': self.signalStack, 'bpm': self.bpm, 'most_periodic': self.most_periodic})

            # Set the 'previous' image to be the current one
            # and the previous point positions to be the current ones
            old_img = new_img.copy()
            p0 = good_new.reshape(-1, 1, 2)

    def MeasureMovingPoints(self, iterator):
        # create a butterworth filter
        butter_filter = signals.make_filter(sample_freq=250)
        # initialise a variable for the signal to go in
        self.signalStack = np.ndarray((0, self.n_components))
        # Set frequency to interpolate to
        sample_freq = 250.0
        # Track some points in a video, changing over time
        for data in signals.window(iterator, self.windowSize, self.windowSkip):

            points = np.array([p[0] for p in data])
            fps = np.mean([p[1] for p in data])

            # Interpolate the points to 250 Hz

            interpolated = signals.interpolate_points(np.vstack(points), fps=fps, sample_freq = sample_freq).T

            # Filter unstable movements
            # interpolated = filter_unstable_movements(interpolated.T).T

            # Filter with a butterworth filter
            filtered = butter_filter(interpolated).T

            # For fitting PCA, remove the time-frames with the top 25% percentile
            # largfest mvoements
            norms = np.linalg.norm(filtered, 2, axis=1)
            removed_abnormalities = filtered[norms > np.percentile(norms, 75)]

            # Perform PCA, getting the largest 5 components of movement
            if self.incrementalPCA:
                self.pca.partial_fit(removed_abnormalities)
            else:
                self.pca.fit(removed_abnormalities)

            # Project the tracked point movements on to the principle component vectors,
            # producing five waveforms for the different components of movement
            transformed = self.pca.transform(filtered)

            self.signalStack = np.vstack((self.signalStack, transformed))

            # Find the periodicity of each signal
            frequencies, periodicities = signals.find_periodicities(self.signalStack)

            # Find the indices of peaks in the signal
            peaks = [list(signals.getpeaks(self.signalStack.T[i]))
                     for i in range(5)]
            most_periodic = np.argmax(periodicities)
            self.most_periodic = most_periodic
            # The frequency of the most periodic signal is supposedly the heart
            # rate
            print "Periodicities: ", periodicities
            print "Most periodic: ", most_periodic
            print "Frequencies: ", frequencies
            print "Peak count BPMs: ", [len(p) for p in peaks]
            print "Heart rate by FFT estimate: {} BPM".format(60.0 * frequencies[most_periodic])
            num_peaks = len(peaks[most_periodic])
            num_seconds = len(self.signalStack) / (sample_freq)
            countbpm = num_peaks * (60.0 / num_seconds)
            print "Heart rate by peak estimate: {} BPM".format(countbpm)

            self.bpm = countbpm

            if False:
                with open("all.pkl", "w") as f:
                    pickle.dump({
                        'frequencies': frequencies, 
                        'periodicities': periodicities,
                        'points': points,
                        'interpolated': interpolated,
                        'filtered': filtered,
                        'transformed': transformed,
                        'stack': self.signalStack,
                        'peaks': peaks,
                        'fftbpm': (60.0 *  frequencies[most_periodic])
                    }, f)


def main():
    """ Run the full algorithm on a video """

    file_params_vstream = dict(
        maxFrameNumber = 300,
        sourceType = "File",
        fileName = "face2-2.mp4",
        maxFPS = None,
        printSteps = True
    )

    file_params_frameproc = dict(
        n_components = 5,
        n_trackPoints = 50,
        windowSize = 30,
        drawFaceTrack = False,
        incrementalPCA = False
    )

    webcam_params_vstream = dict(
        maxFrameNumber = 10000,
        sourceType = "Webcam",
        fileName = "",
        maxFPS = None,
        printSteps = True
    )
    webcam_params_frameproc = dict(
        n_components = 5,
        n_trackPoints = 50,
        windowSize = 10,
        drawFaceTrack = True,
        printSteps = True,
        incrementalPCA = False
    )

    if True:
        frameproc_params = webcam_params_frameproc
        vstream_params = webcam_params_vstream
    else:
        frameproc_params = file_params_frameproc
        vstream_params = file_params_vstream


    
    frameproc_params['windowSkip'] = frameproc_params['windowSize'] - 1


    # Create a video stream producer
    vstream = VideoStreamProducer(**vstream_params)

    # Create a video stream processor
    frameproc = FrameProcessor(vstream.FrameQueue, vstream.pointsQueue, vstream.dataQueue, **frameproc_params)

    # Set the video processor going in its own thread (it will wait until it has
    #   something to process)
    frameproc.start()

    # Set the frame producer going (this will block until the frame processor
    #   has finished)
    vstream.ProduceFrames()



if __name__ == '__main__':
    main()
