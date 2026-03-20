from typing import Optional, List, Tuple
import cv2
import numpy as np

class Ball2D:
    def __init__(self, x: float, y: float, radius: float, confidence: float) -> None:
        self.x = x
        self.y = y
        self.radius = radius
        self.confidence = confidence

class BallDetector:
    def __init__(self, mog2: Optional[cv2.BackgroundSubtractor] = None):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.mog2 = mog2 if mog2 else cv2.createBackgroundSubtractorMOG2()

    def detect(self, frame: np.ndarray, s_max: int = 80, v_min: int = 20, p1: int = 100, p2: int = 20, roi_polygon: Optional[List[Tuple[int, int]]] = None, use_mog2: bool = True, min_radius: float = 15.0, max_radius: float = 35.0) -> Tuple[Optional[Ball2D], np.ndarray]:
        h, w = frame.shape[:2]

        # 1. ROI Bounding Box (für Performance-Boost)
        roi_x, roi_y, roi_w, roi_h = 0, 0, w, h
        roi_mask = None

        if roi_polygon and len(roi_polygon) >= 3:
            pts = np.array(roi_polygon, dtype=np.int32)
            roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(pts)
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [pts], 255)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. Farbmaske
        lower = np.array([0, 0, v_min], dtype=np.uint8)
        upper = np.array([180, s_max, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # 3. MOG2 ROI-Anwendung (für Performance)
        if use_mog2 and self.mog2 is not None:
            roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            fg_mask_roi = self.mog2.apply(roi_frame)

            fg_mask = np.zeros((h, w), dtype=np.uint8)
            fg_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = fg_mask_roi

            mask = cv2.bitwise_and(mask, fg_mask)

        if roi_mask is not None:
            mask = cv2.bitwise_and(mask, roi_mask)

        # 4. Rauschen entfernen (Morphologische Operationen)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel, iterations=2)

        # 5. Konturen finden
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_ball = None
        best_radius = 0

        for cnt in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            if min_radius < radius < max_radius:
                if radius > best_radius:
                    best_radius = radius
                    best_ball = Ball2D(x=float(x), y=float(y), radius=float(radius), confidence=1.0)

        return best_ball, mask