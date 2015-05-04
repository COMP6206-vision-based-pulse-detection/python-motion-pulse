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
import threading
import Queue
import time


class VideoStreamProducer:
    def __init__(self, maxFrameNumber, sourceType, fileName=None,
                 maxFPS=None, printSteps=False):
        self.frameQueue = Queue.Queue(maxFrameNumber)
        self.maxFrameNumber = maxFrameNumber
        self.sourceType = sourceType
        if self.sourceType=="Webcam":
            self.sourceName=0
        elif self.sourceType=="File":
            if fileName is None:
                raise Exception("Filename not specified")
            else:
                self.sourceName=fileName
        else:
            raise Exception("Source type not specified ('Webcam' or 'File')")
        self.maxFPS = maxFPS
        self.printSteps = printSteps
        if self.printSteps:
            print "VideoStreamProducer created. Source = "+self.sourceType+\
                  (" '"+self.sourceName+"'") if self.sourceType=="File" else ""
    @property
    def FrameQueue(self):
        return self.frameQueue
    def ProduceFrames(self):
        """This function blocks until maxFrameNumber frames are produced"""
        if self.printSteps:
            print "VideoStreamProducer opening video source"
        camera = cv2.VideoCapture(self.sourceName)
        #if we are reading from a webcam, we need to calculate FPS manually,
        #   and for that we need to measure time differences
        if self.sourceType == "Webcam":
            oldTime=time.time()
        #go through all the frames we need to collect
        for i in range(self.maxFrameNumber):
            #see if there is a maximum frame period - if so, wait for that
            #   long. The reason we do this before we read & put the first
            #   frame is that it means the first FPS measurement is at least
            #   aproximately right
            if not (self.maxFPS is None):
                time.sleep(1.0/self.maxFPS)
            #read from the source
            success, frame = camera.read()
            #if there was an error, break the loop
            if not success:
                print "Video stream ended unexpectedly"
                break
            #measure the fps
            if self.sourceType=="Webcam":
                currentTime = time.time()
                fps = 1.0/(currentTime-oldTime)
                oldTime=currentTime
            elif self.sourceType=="File":
                fps = camera.get(cv2.cv.CV_CAP_PROP_FPS)
            #print if necessary
            if self.printSteps:
                print "Put frame "+str(i)+" into queue. FPS = "+str(fps)
            #put the frame into the queue
            self.frameQueue.put((frame, fps))
        if self.printSteps:
            print "VideoStreamProducer finished loading frames; getting ready to join threads"
        self.frameQueue.join()

class FrameProcessor(threading.Thread):
    def __init__(self,frameQueue,incrementalPCA,n_components,drawFaceTrack,
                 n_trackPoints,windowSize,windowSkip,printSteps=False):
        threading.Thread.__init__(self)
        self.daemon = True
        self.frameQueue = frameQueue
        self.incrementalPCA = incrementalPCA
        if self.incrementalPCA:
            self.pca = sklearn.decomposition.IncrementalPCA(n_components=n_components)
        else:
            self.pca = sklearn.decomposition.PCA(n_components=n_components)
        self.drawFaceTrack = drawFaceTrack
        self.n_trackPoints = n_trackPoints
        self.windowSize = windowSize
        self.windowSkip = windowSkip
        self.printSteps = printSteps
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
        feature_params = dict(maxCorners=self.n_trackPoints,
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
        # Build a mask which covers a detected face, except for the eys
        faces = ()
        print " *** Searching for a face... *** "
        while len(faces) == 0:
            frame = self.frameQueue.get(block=True)[0]
            old_img = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
            if self.drawFaceTrack:
                cv2.imshow("Video", old_img)
                cv2.waitKey(1)
            faces = face_cascade.detectMultiScale(old_img, 1.3, 5)
            self.frameQueue.task_done()
            
        print " *** Found face *** "
        # Build a mask which covers a detected face, except for the eyes
        rects = make_face_rects(faces[0])
        mask = np.zeros_like(old_img)
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
            if self.drawFaceTrack:
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.line(old_img, (a, b), (c, d), color[i].tolist(), 2)
                    cv2.circle(old_img, (a, b), 5, color[i].tolist(), -1)
                cv2.imshow("Video", old_img)
                cv2.waitKey(1)

            # Yield the y-component of the point positions, and the fps
            yield ((good_new - good_first)[:, 1], fps)

            #say that we've finished with the frame
            self.frameQueue.task_done()

            # Set the 'previous' image to be the current one
            # and the previous point positions to be the current ones
            old_img = new_img.copy()
            p0 = good_new.reshape(-1, 1, 2)

    def MeasureMovingPoints(self,iterator):
        #create a butterworth filter
        butter_filter = make_filter()
        #initialise a variable for the signal to go in
        signalStack = np.ndarray((0, 5))
        # Track some points in a video, changing over time
        for data in window(iterator, self.windowSize, self.windowSkip):

            points = np.array([p[0] for p in data])
            fps = np.mean([p[1] for p in data])

            # Interpolate the points to 250 Hz

            interpolated = interpolate_points(np.vstack(points), fps=fps).T

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

            signalStack = np.vstack((signalStack, transformed))

            # Find the periodicity of each signal
            frequencies, periodicities = find_periodicities(signalStack)

            # Find the indices of peaks in the signal
            peaks = [list(getpeaks(signalStack.T[i])) for i in range(5)]
            most_periodic = np.argmax(periodicities)

            # The frequency of the most periodic signal is supposedly the heart
            # rate
            print "Periodicities: ", periodicities
            print "Most periodic: ", most_periodic
            print "Frequencies: ", frequencies
            print "Peak count BPMs: ", [len(p) for p in peaks]
            print "Heart rate by FFT estimate: {} BPM".format(60.0 / frequencies[most_periodic])
            countbpm = 10.0 / \
                ((signalStack.shape[0] / (250.0 / 30.0)) / 30) * \
                6 * len(peaks[most_periodic])
            print signalStack.shape
            print "Heart rate by peak estimate: {} BPM".format(countbpm)


def main():
    """ Run the full algorithm on a video """

    #Maximum number of frames to load from source
    maxFrameNumber = 300
    #What the video source is (either "File" or "Webcam")
    videoSourceType = "File"
    #Filename of video (if taken from a file)
    videoFileName = "face2-2.mp4"

    #How long to wait between loading frames (probably None if loading from a
    #   file, and probably ~30 if using a webcam)
    maxFPS = None

    #Whether to use an incremental PCA or not
    incrementalPCA = False
    #Number of components for the PCA
    n_components = 5
    #Number of points on the face to track
    n_trackPoints = 50
    #Window size (probably ~60 for an incremental PCA, or (maxFrameNumber-1) for
    #   a video file)
    windowSize = 299
    #How far apart subsequent windows should be
    windowSkip = windowSize-1
    
    #Whether to show the video and tracking points as it's being processed
    drawFaceTrack = True
    #Whether to print out what's being done at each step
    printSteps = True

    #Create a video stream producer
    vstream = VideoStreamProducer(maxFrameNumber,videoSourceType,videoFileName,
                                  maxFPS,printSteps)

    #Create a video stream processor
    frameproc = FrameProcessor(vstream.FrameQueue,incrementalPCA,n_components,
                               drawFaceTrack,n_trackPoints,windowSize,
                               windowSkip,printSteps)

    #Set the video processor going in its own thread (it will wait until it has
    #   something to process)
    frameproc.start()

    #Set the frame producer going (this will block until the frame processor
    #   has finished)
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
    main()

