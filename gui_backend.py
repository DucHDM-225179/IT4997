import av
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QWaitCondition
from PyQt6.QtGui import QImage

class VideoDecoderThread(QThread):
    # Emits (image, frame_number, timestamp_sec)
    frameReady = pyqtSignal(QImage, int, float)
    
    def __init__(self):
        super().__init__()
        self.container = None
        self.video_stream = None
        self._is_running = False
        self._is_playing = False
        self._abort = False
        self._seek_requested = False
        self._seek_frame = -1
        
        self.mutex = QMutex()
        self.cond = QWaitCondition()
        
        self.fps = 30.0
        self.total_frames = 0
        self.duration_sec = 0.0
        self.pts_map = []
        
        self.current_frame_idx = -1
        self.current_time_sec = 0.0

    def open_video(self, path):
        with QMutexLocker(self.mutex):
            if self.container:
                self.container.close()
            
            try:
                # ignore_editlist prevents PyAV from dropping the starting sequence of frames 
                # (which have negative PTS) as dictated by some MP4 edit lists.
                self.container = av.open(path, container_options={'ignore_editlist': '1'})
                self.video_stream = self.container.streams.video[0]
                self.video_stream.thread_type = "AUTO"  # enable multi-threading decoding
                
                # Build PTS map for frame-accurate seeking and true frame count
                # We use decode() instead of demux() because some packets don't yield frames
                # and decoded frames have properly adjusted PTS values.
                pts_set = set()
                for f in self.container.decode(video=0):
                    if f.pts is not None:
                        pts_set.add(f.pts)
                self.pts_map = sorted(list(pts_set))
                
                # Reset container after demux pass
                self.container.seek(-1, backward=True)
                
                # Calculate FPS and duration
                self.fps = float(self.video_stream.average_rate)
                if self.fps <= 0:
                    self.fps = 30.0
                    
                self.total_frames = len(self.pts_map)
                self.duration_sec = float(self.video_stream.duration * self.video_stream.time_base) if self.video_stream.duration else (self.total_frames / self.fps)
                
                self.current_frame_idx = -1
                self.current_time_sec = 0.0
                return True
            except Exception as e:
                print(f"Error opening video: {e}")
                self.container = None
                self.video_stream = None
                return False

    def get_metadata(self):
        with QMutexLocker(self.mutex):
            return {
                "fps": self.fps,
                "total_frames": self.total_frames,
                "duration_sec": self.duration_sec,
                "width": self.video_stream.width if self.video_stream else 0,
                "height": self.video_stream.height if self.video_stream else 0
            }

    def play(self):
        with QMutexLocker(self.mutex):
            self._is_playing = True
            self.cond.wakeAll()

    def pause(self):
        with QMutexLocker(self.mutex):
            self._is_playing = False

    def seek_frame(self, frame_num):
        with QMutexLocker(self.mutex):
            self._seek_requested = True
            self._seek_frame = max(0, min(frame_num, self.total_frames - 1))
            self.cond.wakeAll()
            
    def step_forward(self):
        with QMutexLocker(self.mutex):
            self._seek_requested = True
            self._seek_frame = min(self.current_frame_idx + 1, self.total_frames - 1)
            self.cond.wakeAll()
            
    def step_backward(self):
        with QMutexLocker(self.mutex):
            self._seek_requested = True
            self._seek_frame = max(0, self.current_frame_idx - 1)
            self.cond.wakeAll()

    def stop(self):
        with QMutexLocker(self.mutex):
            self._abort = True
            self.cond.wakeAll()
        self.wait()
        if self.container:
            self.container.close()

    def run(self):
        self._is_running = True
        internal_frame_idx = -1
        
        while True:
            with QMutexLocker(self.mutex):
                if self._abort:
                    break
                
                if not self.container or (not self._is_playing and not self._seek_requested):
                    self.cond.wait(self.mutex)
                    continue
                    
                seek_req = self._seek_requested
                seek_target = self._seek_frame
                is_playing = self._is_playing
                
                if seek_req:
                    self._seek_requested = False
            
            try:
                if seek_req:
                    # Get exact PTS from our map
                    target_pts = self.pts_map[seek_target] if self.pts_map else 0
                    
                    # Seek to keyframe before or at target_pts
                    self.container.seek(target_pts, stream=self.video_stream, backward=True)
                    
                    # Read until we hit the exact frame (or very close)
                    found_frame = None
                    for frame in self.container.decode(video=0):
                        if self._abort:
                            return
                        found_frame = frame
                        # We allow a small tolerance or exactly matching index
                        if frame.pts >= target_pts:
                            break
                            
                    if found_frame:
                        internal_frame_idx = seek_target
                        self._emit_frame(found_frame, seek_target)
                        # Sleep briefly to allow UI to process if seeking rapidly
                        time.sleep(0.01)
                
                elif is_playing:
                    start_time = time.time()
                    
                    try:
                        packet = next(self.container.demux(video=0))
                        for frame in packet.decode():
                            if self._abort:
                                return
                            
                            internal_frame_idx += 1
                            if internal_frame_idx >= self.total_frames:
                                internal_frame_idx = self.total_frames - 1
                                
                            self._emit_frame(frame, internal_frame_idx)
                            
                            # Maintain FPS per frame
                            elapsed = time.time() - start_time
                            delay = (1.0 / self.fps) - elapsed
                            if delay > 0:
                                time.sleep(delay)
                            start_time = time.time()
                            
                    except StopIteration:
                        # End of stream
                        with QMutexLocker(self.mutex):
                            self._is_playing = False
                        continue
                        
            except Exception as e:
                print(f"Decoder error: {e}")
                with QMutexLocker(self.mutex):
                    self._is_playing = False

    def _emit_frame(self, frame, frame_idx):
        # Convert PyAV VideoFrame to QImage
        img_array = frame.to_ndarray(format='rgb24')
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        # QImage needs bytes, so we use tobytes()
        qimg = QImage(img_array.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        
        current_sec = float(frame.pts * self.video_stream.time_base)
        
        with QMutexLocker(self.mutex):
            self.current_frame_idx = frame_idx
            self.current_time_sec = current_sec
            
        self.frameReady.emit(qimg, frame_idx, current_sec)
