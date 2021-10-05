from abc import ABC, abstractmethod
from functools import partial
import multiprocessing as mp
import queue
import signal
import sys
import time
from typing import Optional, Union

import cv2


class FrameReaderMgrBase(ABC):
    def __init__(
        self,
        *,
        frame_timeout: Optional[int] = 5,
        queue_length: Optional[int] = 64,
        frame_cutoff: Optional[int] = float("inf"),
        frame_formatters: Optional[int] = 8,
    ):
        """
        This is the abstract base class for reading frames from a video file. It starts one process
        for reading directly from the video and a configurable number of processes formatting each
        frame. To use, subclass this and implement the _format_img method, then use it as a context
        manager. For example:

            frame_reader = FrameReaderMgrSubclass()
            with frame_reader(video_path):
                while True:
                    try:
                        images, frames = frame_reader.get_frames(batch_size)
                    except:
                        break
                    # Do something with the image here
        """
        self._frame_timeout = frame_timeout
        self._frame_cutoff = frame_cutoff
        self._raw_queue = mp.Queue(queue_length)
        self._frame_queue = mp.Queue(queue_length)
        self._stop_event = mp.Event()
        self._frame_stop_event = mp.Event()
        self._done_event = mp.Event()
        self._read_frames_process = None
        self._enqueue_frames_processes = [None for _ in range(frame_formatters)]
        self._signal = signal.signal(signal.SIGINT, partial(self._signal_handler))

    def get_frames(self, n: int):
        """
        The interface for getting frames from the frame reader. n is the desired number of frames to
        return from this function. It will return up to n frames, if there are any left to return,
        or it will raise a `queue.Empty` exception if the frame queue is empty. Returns a list of
        elements whose type is determined by the implementation of _format_img.
        """
        try:
            # The first .get() is outside the while loop to notify the caller that there are no
            # frames left to get and to set the done event
            result = [self._frame_queue.get(timeout=self._frame_timeout)]
        except queue.Empty:
            self._done_event.set()
            raise

        while len(result) < n:
            # .get() more until `self._frame_queue` is empty or `n` items obtained
            try:
                result.append(self._frame_queue.get(timeout=self._frame_timeout))
            except queue.Empty:
                break

        return result

    def stop_reading(self):
        """
        Sets the appropriate stop events to signal the processes.
        """
        self._stop_event.set()
        self._frame_stop_event.set()

    @staticmethod
    def _terminate_if_alive(process):
        if process and process.is_alive():
            process.terminate()

    def __call__(self, vid_path):
        # Clear out queues
        while not self._raw_queue.empty():
            self._raw_queue.get_nowait()
        while not self._frame_queue.empty():
            self._frame_queue.get_nowait()

        # Clear Events
        self._stop_event.clear()
        self._frame_stop_event.clear()
        self._done_event.clear()

        # Terminate old processes
        self._terminate_if_alive(self._read_frames_process)
        for process in self._enqueue_frames_processes:
            self._terminate_if_alive(process)

        # Start frame processors
        self._read_frames_process = mp.Process(target=self._read_frames, args=(vid_path,))
        self._read_frames_process.daemon = True
        self._read_frames_process.start()

        for idx in range(len(self._enqueue_frames_processes)):
            p = mp.Process(target=self._enqueue_frames)
            p.daemon = True
            p.start()
            self._enqueue_frames_processes[idx] = p

        return self

    def __enter__(self):
        """
        The heavy-lifting is done by __call__, which allows this class to be used as a context
        manager or manually managed, similar to the built-in `open()`. If used manually, call
        stop_reading to close all files and processes.
        """
        pass

    def __exit__(self, typ, value, traceback):
        self.stop_reading()

    def _signal_handler(self, signal_received, frame):
        self.__exit__(None, None, None)
        sys.exit(0)

    def _read_frames(self, vid_path):
        """
        Reads frames from the video and puts them in the raw queue.
        """
        ok = True
        vid = cv2.VideoCapture(vid_path)
        frame_num = 0
        while ok and not self._stop_event.is_set():
            if not self._raw_queue.full():
                ok, img = vid.read()
                if ok:
                    self._raw_queue.put((img, frame_num))
                    frame_num += 1
            else:
                time.sleep(0.1)
            if frame_num > self._frame_cutoff:
                break
        self._done_event.wait()
        self._stop_event.set()
        vid.release()

    @abstractmethod
    def _format_img(self, img, frame_num):
        """
        Applies a generic transform to the image. Input is a numpy array and the output is the
        desired element to enqueue in the frame queue.
        """

    def _enqueue_frames(self):
        """
        Reads frames from the raw queue and applies the image formatter before putting them in the
        processed frame queue.
        """
        while not self._stop_event.is_set():
            try:
                img, frame_num = self._raw_queue.get(timeout=self._frame_timeout)
                X = self._format_img(img, frame_num)
                self._frame_queue.put(X)
            except:
                self._done_event.wait()
                self._stop_event.set()
