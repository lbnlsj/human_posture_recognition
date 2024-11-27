import cv2
import numpy as np
from pathlib import Path
import torch
import time
from typing import List, Tuple, Dict
from collections import deque
import threading


class CameraConfig:
    def __init__(self):
        self.NUM_CAMERAS = 3
        self.CAMERA_FOV = 120
        self.TOTAL_FOV = 200
        self.FRAME_WIDTH = 1280
        self.FRAME_HEIGHT = 720
        self.FPS = 15
        self.CAMERA_SPACING = 30
        self.MOUNTING_HEIGHT = 250
        self.OVERLAP_PERCENT = 0.2
        self.NIGHT_MODE_THRESHOLD = 50


class FenceManager:
    def __init__(self):
        self.fences = []
        self.alerts = []

    def add_fence(self, points: List[Tuple[int, int]], name: str):
        self.fences.append({
            'points': np.array(points),
            'name': name,
            'contour': cv2.convexHull(np.array(points))
        })

    def check_position(self, point: Tuple[int, int]) -> str:
        for fence in self.fences:
            if cv2.pointPolygonTest(fence['contour'], point, False) >= 0:
                return fence['name']
        return None


class PersonTracker:
    def __init__(self, max_disappeared=30):
        self.tracks = {}
        self.next_id = 0
        self.max_disappeared = max_disappeared

    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection / float(box1_area + box2_area - intersection)

    def update(self, detections: List[Dict]):
        current_boxes = {i: det['bbox'] for i, det in enumerate(detections)}
        matched_tracks = []

        # Match existing tracks
        for track_id, track in self.tracks.items():
            if track['active']:
                best_iou = 0
                best_detection = None

                for det_id, box in current_boxes.items():
                    iou = self._calculate_iou(track['bbox'], box)
                    if iou > best_iou:
                        best_iou = iou
                        best_detection = det_id

                if best_iou > 0.3:
                    matched_tracks.append((track_id, best_detection))
                    del current_boxes[best_detection]
                else:
                    track['disappeared'] += 1

        # Update tracks
        for track_id, det_id in matched_tracks:
            det = detections[det_id]
            self.tracks[track_id].update({
                'bbox': det['bbox'],
                'position': det['position'],
                'disappeared': 0,
                'last_seen': time.time()
            })

        # Add new tracks
        for det_id in current_boxes:
            det = detections[det_id]
            self.tracks[self.next_id] = {
                'bbox': det['bbox'],
                'position': det['position'],
                'disappeared': 0,
                'active': True,
                'last_seen': time.time(),
                'trajectory': deque(maxlen=30)
            }
            self.next_id += 1

        # Remove old tracks
        self._cleanup_tracks()

    def _cleanup_tracks(self):
        current_time = time.time()
        to_delete = []

        for track_id, track in self.tracks.items():
            if track['disappeared'] > self.max_disappeared:
                to_delete.append(track_id)

        for track_id in to_delete:
            del self.tracks[track_id]


class MultiCameraStitcher:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cameras = []
        self.frame_buffers = []
        self.homography_matrices = []
        self.fence_manager = FenceManager()
        self.person_tracker = PersonTracker()

        self.initialize_cameras()
        self.calibrate_cameras()
        self._setup_model()

        # Initialize frame buffers
        for _ in range(self.config.NUM_CAMERAS):
            self.frame_buffers.append(deque(maxlen=5))

        # Start capture threads
        self.capture_threads = []
        self.running = True
        self._start_capture_threads()

    def _setup_model(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.classes = [0]  # 只检测人体
        if torch.cuda.is_available():
            self.model.cuda()

    def _capture_frames(self, camera_idx):
        while self.running:
            ret, frame = self.cameras[camera_idx].read()
            if ret:
                self.frame_buffers[camera_idx].append(frame)
            time.sleep(1 / self.config.FPS)

    def _start_capture_threads(self):
        for i in range(self.config.NUM_CAMERAS):
            thread = threading.Thread(target=self._capture_frames, args=(i,))
            thread.daemon = True
            thread.start()
            self.capture_threads.append(thread)

    def enhance_night_vision(self, frame):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        yuv = cv2.GaussianBlur(yuv, (5, 5), 0)
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)

    def process_frame(self, night_mode: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        frames = []
        for buffer in self.frame_buffers:
            if buffer:
                frame = buffer[-1].copy()
                if night_mode:
                    frame = self.enhance_night_vision(frame)
                frames.append(frame)

        if not frames:
            raise RuntimeError("No frames available")

        stitched = self.stitch_frames(frames)
        bird_eye = self.get_bird_eye_view(stitched)
        detections = self.detect_humans(stitched)
        self.person_tracker.update(detections)

        # Draw detections and tracks
        for track_id, track in self.person_tracker.tracks.items():
            if track['active']:
                pos = track['position']
                cv2.circle(bird_eye, (int(pos[0]), int(pos[1])), 5, (0, 255, 0), -1)
                cv2.putText(bird_eye, f"ID: {track_id}",
                            (int(pos[0]), int(pos[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check fence violations
                fence_name = self.fence_manager.check_position(pos)
                if fence_name:
                    cv2.putText(bird_eye, f"In {fence_name}",
                                (int(pos[0]), int(pos[1]) - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return bird_eye, detections

    def run(self):
        try:
            while True:
                start_time = time.time()

                try:
                    # Auto detect night mode based on average brightness
                    sample_frame = self.frame_buffers[0][-1] if self.frame_buffers[0] else None
                    if sample_frame is not None:
                        avg_brightness = np.mean(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY))
                        night_mode = avg_brightness < self.config.NIGHT_MODE_THRESHOLD

                        frame, detections = self.process_frame(night_mode)

                        # Add status overlay
                        status_text = f"FPS: {1.0 / (time.time() - start_time):.1f}"
                        status_text += f" | Mode: {'Night' if night_mode else 'Day'}"
                        status_text += f" | Tracks: {len(self.person_tracker.tracks)}"

                        cv2.putText(frame, status_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        cv2.imshow('Bird Eye View', frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

        finally:
            self.running = False
            for thread in self.capture_threads:
                thread.join()
            for cap in self.cameras:
                cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    config = CameraConfig()
    stitcher = MultiCameraStitcher(config)

    # 添加示例电子围栏
    stitcher.fence_manager.add_fence([
        (100, 100), (300, 100),
        (300, 300), (100, 300)
    ], "Zone A")

    stitcher = MultiCameraStitcher(config)

    # 添加示例电子围栏
    stitcher.fence_manager.add_fence([
        (100, 100), (300, 100),
        (300, 300), (100, 300)
    ], "Zone A")

    stitcher.fence_manager.add_fence([
        (400, 200), (600, 200),
        (600, 400), (400, 400)
    ], "Zone B")

    # 运行系统
    stitcher.run()
