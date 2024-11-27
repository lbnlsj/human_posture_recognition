import cv2
import numpy as np
from pathlib import Path
import torch
import time
from typing import List, Tuple, Dict


class CameraConfig:
    def __init__(self):
        self.NUM_CAMERAS = 3
        self.CAMERA_FOV = 120  # 每个摄像头视角
        self.TOTAL_FOV = 200  # 期望合成视角
        self.FRAME_WIDTH = 1280
        self.FRAME_HEIGHT = 720
        self.FPS = 15


class MultiCameraStitcher:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cameras = []
        self.homography_matrices = []
        self.initialize_cameras()
        self.calibrate_cameras()

        # 加载YOLOv5模型用于人体检测
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.classes = [0]  # 只检测人体
        if torch.cuda.is_available():
            self.model.cuda()

    def initialize_cameras(self):
        """初始化摄像头"""
        for i in range(self.config.NUM_CAMERAS):
            cap = cv2.VideoCapture(i)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, self.config.FPS)
            self.cameras.append(cap)

    def calibrate_cameras(self):
        """计算相机之间的单应性矩阵"""
        # 为简化示例，这里使用预定义的变换矩阵
        # 实际应用中需要通过特征点匹配计算
        for i in range(self.config.NUM_CAMERAS - 1):
            H = np.eye(3)
            H[0, 2] = self.config.FRAME_WIDTH * 0.8  # 假设80%重叠
            self.homography_matrices.append(H)

    def get_bird_eye_view_matrix(self):
        """计算鸟瞰图变换矩阵"""
        src_points = np.float32([[0, 0], [self.config.FRAME_WIDTH, 0],
                                 [0, self.config.FRAME_HEIGHT],
                                 [self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT]])
        dst_points = np.float32([[100, 0], [self.config.FRAME_WIDTH - 100, 0],
                                 [0, self.config.FRAME_HEIGHT],
                                 [self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT]])
        return cv2.getPerspectiveTransform(src_points, dst_points)

    def stitch_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """拼接多个摄像头画面"""
        result = frames[0]
        for i in range(len(frames) - 1):
            next_frame = frames[i + 1]
            H = self.homography_matrices[i]
            warped = cv2.warpPerspective(next_frame, H,
                                         (result.shape[1] + int(self.config.FRAME_WIDTH * 0.2),
                                          result.shape[0]))
            result = self.blend_images(result, warped)
        return result

    def blend_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """平滑混合两张图片"""
        overlap = 100  # 重叠区域宽度
        mask = np.zeros_like(img1)
        mask[:, -overlap:] = np.linspace(1, 0, overlap).reshape(1, -1, 1)
        return img1 * (1 - mask) + img2 * mask

    def detect_humans(self, frame: np.ndarray) -> List[Dict]:
        """使用YOLOv5检测人体"""
        results = self.model(frame)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.5:  # 置信度阈值
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'position': self.get_ground_position([x1, y1, x2, y2])
                })
        return detections

    def get_ground_position(self, bbox: List[int]) -> Tuple[float, float]:
        """计算人体在地面上的位置"""
        foot_point = [(bbox[0] + bbox[2]) // 2, bbox[3]]
        matrix = self.get_bird_eye_view_matrix()
        point = cv2.perspectiveTransform(np.array([[foot_point]], dtype='float32'), matrix)
        return tuple(point[0][0])

    def process_frame(self, night_mode: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        """处理一帧图像"""
        frames = []
        for cap in self.cameras:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame")

            if night_mode:
                # 夜视模式增强
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.equalizeHist(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            frames.append(frame)

        # 拼接图像
        stitched = self.stitch_frames(frames)

        # 鸟瞰图变换
        M = self.get_bird_eye_view_matrix()
        bird_eye = cv2.warpPerspective(stitched, M,
                                       (stitched.shape[1], stitched.shape[0]))

        # 人体检测
        detections = self.detect_humans(stitched)

        # 在鸟瞰图上标注检测结果
        for det in detections:
            pos = det['position']
            cv2.circle(bird_eye, (int(pos[0]), int(pos[1])), 5, (0, 255, 0), -1)

        return bird_eye, detections

    def run(self):
        """主循环"""
        try:
            while True:
                start_time = time.time()

                try:
                    frame, detections = self.process_frame()
                    cv2.imshow('Bird Eye View', frame)

                    # 控制帧率
                    elapsed = time.time() - start_time
                    wait_time = max(1, int((1.0 / self.config.FPS - elapsed) * 1000))
                    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                        break

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

        finally:
            for cap in self.cameras:
                cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    config = CameraConfig()
    stitcher = MultiCameraStitcher(config)
    stitcher.run()