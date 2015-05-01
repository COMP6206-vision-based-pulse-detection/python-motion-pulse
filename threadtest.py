import threading
import time
import Queue


class Producer:
    def __init__(self, maxNumber, frameQueue):
        self.maxNumber = maxNumber
        self.frameQueue = frameQueue
    def Produce(self):
        for i in range(self.maxNumber):
            time.sleep(1)
            print "Producing frame: " + str(i)
            self.frameQueue.put(i)

class Consumer(threading.Thread):
    def __init__(self,frameQueue):
        threading.Thread.__init__(self)
        self.daemon=True
        self.frameQueue = frameQueue
    def run(self):
        self.ProcessFrames()
    def ProcessFrames(self):
        while True:
            frame=self.frameQueue.get(block=True)
            print "Processing frame: " + str(frame)
            self.frameQueue.task_done()

q=Queue.Queue(100)

#create thread
Sim = Consumer(q)
Sim.start()

Dave = Producer(10,q)
Dave.Produce()

q.join()

print "Exiting Main Thread"

