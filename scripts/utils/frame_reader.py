from abc import ABC, abstractmethod


class FrameReaderMgrBase(ABC):
    def __init__(
        self,
        queue_length: Optional[int] = 64,
        frame_cutoff: Optional[int] = float("inf"),
        frame_formatters: Optional[int] = 8,
    ):
        self._raw_queue = mp.Queue(queue_length)
        self._frame_queue = mp.Queue(queue_length)
        self._stop_event = mp.Event()
        self._frame_stop_event = mp.Event()
        self._done_event = mp.Event()
        self._read_frames_process = None
        self._enqueue_frames_processes = [None for _ in range(frame_formatters)]
        self._signal = signal.signal(signal.SIGINT, partial(self._signal_handler))

    @staticmethod
    def _terminate_if_alive(process: Union[mp.Process, None]):
        if process and process.is_alive():
            process.terminate()

    def __enter__(self, vid_path):
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
        self._read_frames_process = mp.Process(target=self._read_frames, args=(vid_path))
        self._read_frames_process.daemon = True
        self._read_frames_process.start()

        for idx in range(len(self._enqueue_frames_processes)):
            p = mp.Process(target=self._enqueue_frames)
            p.daemon = True
            p.start()
            self._enqueue_frames_processes[idx] = p

        return self._frame_queue, done_event

    def __exit__(self):
        self._stop_event.set()
        self._frame_stop_event.set()

    def _signal_handler(self, signal_received, frame):
        # Handle any cleanup here
        self._stop_event.set()
        self._frame_stop_event.set()
        logger.info("SIGINT or CTRL-C detected. Exiting gracefully")
        for i in range(10):
            logger.info(f"Shutting down in {10-i}")
            time.sleep(1)

    def _read_frames(self, vid_path):
        ok = True
        vid = cv2.VideoCapture(vid_path)
        frame_num = 0
        while ok and not self._stop_event.is_set():
            if not self._raw_queue.full():
                ok, img = vid.read()
                if ok:
                    self._raw_queue.put((img, frame_num))
                    frame_num += 1
            if frame_num > self._frame_cutoff:
                logger.info(f"Reached frame cutoff ({self._frame_cutoff})")
                break
        self._done_event.wait()
        self._stop_event.set()
        vid.release()
        logger.info("This thread should exit now read frames")

    @abstractmethod
    def _format_img(self, img, frame_num):
        """
        Applies a generic transform to the image. Input and output must both be numpy arrays.
        """
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        img = self._augmentation.get_transform(img).apply_image(img)
        _, h, w = img.shape
        return {"image": img, "height": h, "width": w, "frame_num": frame_num}

    def _enqueue_frames(self):
        while not self._stop_event.is_set():
            try:
                img, frame_num = self._raw_queue.get(timeout=FRAME_TIMEOUT)
                X = self._format_img(img, frame_num)
                self._frame_queue.put((X, frame_num))
            except:
                self._done_event.wait()
                self._stop_event.set()
                logger.info("This thread should exit now")
