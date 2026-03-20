from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

import numpy as np

# Konstanten für Skelett-Verbindungen
JOINT_NAMES = [
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]


# --- Datenstrukturen ---

@dataclass
class Pose2D:
    keypoints: Dict[str, np.ndarray]
    confidence: float = 1.0

@dataclass
class Pose3D:
    keypoints: Dict[str, np.ndarray]

@dataclass
class Ball2D:
    center: np.ndarray
    radius: float
    confidence: float = 1.0

@dataclass
class FrameMetrics:
    time_s: float
    trunk_inclination_deg: float
    shoulder_angle_deg: float
    elbow_angle_deg: float
    knee_angle_deg: float
    wrist_speed_mps: float
    wrist_speed_kmph: float

@dataclass
class AnalysisState:
    wrist_history: List[Tuple[float, np.ndarray]] = field(default_factory=list)
    ball_history: List[Tuple[float, np.ndarray]] = field(default_factory=list)
    metrics: List[FrameMetrics] = field(default_factory=list)


# --- Hilfsfunktionen ---

def safe_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def euclidean_distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p2 - p1))

def draw_pose_overlay(frame: np.ndarray, pose2d: Optional[Pose2D]) -> None:
    if pose2d is None:
        return
    for _, p in pose2d.keypoints.items():
        cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)
    for a, b in SKELETON_EDGES:
        pa = pose2d.keypoints.get(a)
        pb = pose2d.keypoints.get(b)
        if pa is not None and pb is not None:
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 200, 0), 2)

def draw_ball_overlay(frame: np.ndarray, ball: Optional[Ball2D]) -> None:
    if ball is None:
        return
    cx, cy = int(ball.center[0]), int(ball.center[1])
    radius = max(1, int(ball.radius))
    cv2.circle(frame, (cx, cy), radius, (0, 90, 255), 2)
    cv2.circle(frame, (cx, cy), 2, (0, 90, 255), -1)

def create_text_panel(text: str, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 40, dtype=np.uint8)
    text = text.replace('°', ' Grad').replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
    text = text.replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue')
    y0, dy = 40, 25
    
    words = text.replace('\n', ' \n ').split(' ')
    lines = []
    current_line = ""
    for word in words:
        if word == '\n':
            lines.append(current_line)
            current_line = ""
        elif len(current_line) + len(word) < 55: 
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "
    if current_line:
        lines.append(current_line)
        
    for i, line in enumerate(lines):
        y = y0 + i * dy
        if y > height - 20:
            break
        cv2.putText(panel, line.strip(), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return panel

def ensure_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) ist nicht installiert. Bitte installiere opencv-python.")

def optional_import(module_name: str):
    try:
        if importlib.util.find_spec(module_name) is None:
            return None
        return importlib.import_module(module_name)
    except Exception:
        return None

def pick_default_pose_backend() -> str:
    mmpose_apis = optional_import("mmpose.apis")
    if mmpose_apis is not None and hasattr(mmpose_apis, "MMPoseInferencer"):
        return "mmpose"
    if sys.version_info < (3, 13):
        mediapipe = optional_import("mediapipe")
        if mediapipe is not None and hasattr(mediapipe, "solutions"):
            return "mediapipe"
    return "none"


# --- Kern-Klassen ---

class Kalman1D:
    def __init__(self, process_var: float = 1e-3, measurement_var: float = 1e-2):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.x = 0.0
        self.p = 1.0
        self.initialized = False

    def update(self, measurement: float) -> float:
        if math.isnan(measurement):
            return self.x
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x
        self.p = self.p + self.process_var
        k = self.p / (self.p + self.measurement_var)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        return self.x

class SkeletonRetargeter:
    def retarget(self, pose3d: Pose3D) -> Pose3D:
        ls = pose3d.keypoints.get("left_shoulder")
        rs = pose3d.keypoints.get("right_shoulder")
        if ls is None or rs is None:
            return pose3d
        shoulder_width = np.linalg.norm(ls - rs)
        if shoulder_width < 1e-6:
            return pose3d
        scale = 1.0 / shoulder_width
        return Pose3D(keypoints={k: v * scale for k, v in pose3d.keypoints.items()})

class UDPoseEstimator:
    def __init__(self, preferred_backend: str = "auto"):
        self.mode = "none"
        self.backend = None
        self.mp_pose = None
        self.preferred_backend = preferred_backend

        if preferred_backend in {"auto", "mmpose"}:
            mmpose_apis = optional_import("mmpose.apis")
            if mmpose_apis is not None and hasattr(mmpose_apis, "MMPoseInferencer"):
                try:
                    self.backend = mmpose_apis.MMPoseInferencer("human")
                except (ModuleNotFoundError, ImportError):
                    pass
                else:
                    self.mode = "udp-mmpose"
                    return

        if preferred_backend in {"auto", "mediapipe"}:
            if sys.version_info < (3, 13):
                mediapipe = optional_import("mediapipe")
                if mediapipe is not None and hasattr(mediapipe, "solutions"):
                    self.mp_pose = mediapipe.solutions.pose.Pose(static_image_mode=False)
                    self.mode = "mediapipe"

    def _estimate_mediapipe(self, frame: np.ndarray, kpt_thr: float) -> Optional[Pose2D]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mp_pose.process(rgb)
        if not result.pose_landmarks:
            return None
        
        h, w = frame.shape[:2]
        lm = result.pose_landmarks.landmark
        idx_map = {
            "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
            "left_elbow": 13, "right_elbow": 14, "left_wrist": 15,
            "right_wrist": 16, "left_hip": 23, "right_hip": 24,
            "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28
        }
        
        keypoints = {}
        for name, idx in idx_map.items():
            if lm[idx].visibility < kpt_thr:
                continue
            keypoints[name] = np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)
            
        if len(keypoints) < 5:
            return None
            
        return Pose2D(keypoints=keypoints, confidence=1.0)

    def estimate(self, frame: np.ndarray, kpt_thr: float = 0.3) -> Optional[Pose2D]:
        if self.mode == "udp-mmpose":
            result_iter = self.backend(frame, return_vis=False)
            result = next(result_iter, None)
            if result is None:
                return None
            preds = result.get("predictions", [])
            keypoints_array = None
            if isinstance(preds, list) and preds:
                first = preds[0]
                if isinstance(first, dict) and "keypoints" in first:
                    keypoints_array = np.array(first.get("keypoints", []), dtype=np.float32)
                elif isinstance(first, list) and first:
                    cand = first[0]
                    if isinstance(cand, dict):
                        for k in ("keypoints", "keypoints_xy", "keypoints_xy_score"):
                            if k in cand:
                                keypoints_array = np.array(cand.get(k, []), dtype=np.float32)
                                break
                    elif isinstance(cand, (list, np.ndarray)):
                        keypoints_array = np.array(cand, dtype=np.float32)
            if keypoints_array is None or keypoints_array.size == 0:
                return None
            
            if keypoints_array.ndim == 1 and keypoints_array.size % 2 == 0:
                keypoints_array = keypoints_array.reshape(-1, 2)
                
            if keypoints_array.shape[1] >= 2:
                if float(np.nanmax(keypoints_array[:, :2])) <= 1.5:
                    h, w = frame.shape[:2]
                    keypoints_array[:, 0] *= w
                    keypoints_array[:, 1] *= h
            
            if keypoints_array.shape[0] < 13:
                return None
            
            coco_map = {
                "nose": 0, "left_shoulder": 5, "right_shoulder": 6,
                "left_elbow": 7, "right_elbow": 8, "left_wrist": 9,
                "right_wrist": 10, "left_hip": 11, "right_hip": 12,
                "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
            }
            
            keypoints = {}
            for name, idx in coco_map.items():
                if idx < keypoints_array.shape[0]:
                    if keypoints_array.shape[1] > 2 and keypoints_array[idx, 2] < kpt_thr:
                        continue
                    keypoints[name] = keypoints_array[idx, :2]
                
            if len(keypoints) < 5:
                return None
            return Pose2D(keypoints=keypoints, confidence=1.0)
            
        if self.mode == "mediapipe":
            mp_pose = self._estimate_mediapipe(frame, kpt_thr)
            if mp_pose is not None:
                return mp_pose

        return None

class BallDetector:
    def __init__(self, mog2: Optional[Any] = None):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.manual_tracker = None
        self.tracker_active = False
        self.color_filter: Optional[Dict[str, Any]] = None
        self.mog2 = mog2
        
        if self.mog2 is None:
            self.reset_background_model()

    def reset_background_model(self) -> None:
        self.mog2 = None
        if cv2 is not None and hasattr(cv2, "createBackgroundSubtractorMOG2"):
            try:
                self.mog2 = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
            except Exception:
                self.mog2 = None

    def init_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        if cv2 is None: return
        try:
            self.manual_tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            try:
                self.manual_tracker = cv2.TrackerMIL_create()
            except AttributeError:
                self.manual_tracker = None
                return
                
        self.manual_tracker.init(frame, bbox)
        self.tracker_active = True

    def reset_tracker(self):
        self.manual_tracker = None
        self.tracker_active = False

    def _pick_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0] if len(contours) == 2 else contours[1]

    def clear_color_filter(self) -> None:
        self.color_filter = None

    def has_color_filter(self) -> bool:
        return self.color_filter is not None

    def _build_hue_ranges(self, center_h: int, tol_h: int) -> List[Tuple[int, int]]:
        lo = (center_h - tol_h) % 180
        hi = (center_h + tol_h) % 180
        if lo <= hi:
            return [(lo, hi)]
        return [(0, hi), (lo, 179)]

    def set_color_filter_from_bgr_samples(
        self,
        bgr_samples: List[Tuple[int, int, int]],
        hue_tolerance: int = 14,
        sat_tolerance: int = 65,
        val_tolerance: int = 65,
    ) -> bool:
        if not bgr_samples:
            self.color_filter = None
            return False

        sample_arr = np.array(bgr_samples, dtype=np.uint8).reshape((-1, 1, 3))
        hsv_samples = cv2.cvtColor(sample_arr, cv2.COLOR_BGR2HSV).reshape((-1, 3)).astype(np.int32)

        median_h = int(np.median(hsv_samples[:, 0]))
        median_s = int(np.median(hsv_samples[:, 1]))
        median_v = int(np.median(hsv_samples[:, 2]))

        h_ranges = self._build_hue_ranges(median_h, max(4, int(hue_tolerance)))
        s_lo = int(max(0, median_s - sat_tolerance))
        s_hi = int(min(255, median_s + sat_tolerance))
        v_lo = int(max(0, median_v - val_tolerance))
        v_hi = int(min(255, median_v + val_tolerance))

        self.color_filter = {
            "h_ranges": h_ranges,
            "s_range": (s_lo, s_hi),
            "v_range": (v_lo, v_hi),
        }
        return True

    def _build_color_mask(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        if not self.color_filter:
            return None

        s_lo, s_hi = self.color_filter["s_range"]
        v_lo, v_hi = self.color_filter["v_range"]
        out_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for h_lo, h_hi in self.color_filter["h_ranges"]:
            lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
            upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
            out_mask = cv2.bitwise_or(out_mask, cv2.inRange(hsv, lower, upper))
        return out_mask

    def detect(self, frame: np.ndarray, s_max: int = 80, v_min: int = 20, p1: int = 100, p2: int = 20, roi_polygon: Optional[List[Tuple[int, int]]] = None, use_mog2: bool = True, min_radius: float = 15.0, max_radius: float = 35.0) -> Tuple[Optional[Ball2D], np.ndarray]:
        h, w = frame.shape[:2]
        
        if self.tracker_active and self.manual_tracker is not None:
            ok, bbox = self.manual_tracker.update(frame)
            if ok:
                x, y, bw, bh = bbox
                center_x = x + bw / 2.0
                center_y = y + bh / 2.0
                radius = max(bw, bh) / 2.0
                return Ball2D(center=np.array([center_x, center_y], dtype=np.float32), radius=float(radius), confidence=1.0), np.zeros((h, w), dtype=np.uint8)
            else:
                self.tracker_active = False
                
        # ROI Bounding Box 
        roi_x, roi_y, roi_w, roi_h = 0, 0, w, h
        roi_mask = None

        if roi_polygon and len(roi_polygon) >= 3:
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(roi_polygon, dtype=np.int32)
            roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(pts)
            cv2.fillPoly(roi_mask, [pts], 255)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        best: Optional[Tuple[float, Ball2D]] = None

        color_mask = self._build_color_mask(hsv)
        if color_mask is not None:
            mask = color_mask
        else:
            lower = np.array([0, 0, v_min], dtype=np.uint8)
            upper = np.array([180, s_max, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

        # MOG2 ROI-Anwendung
        if use_mog2 and self.mog2 is not None:
            fg_mask = self.mog2.apply(frame)
            roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            fg_mask_roi = self.mog2.apply(roi_frame)

            fg_mask = np.zeros((h, w), dtype=np.uint8)
            fg_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = fg_mask_roi

            mask = cv2.bitwise_and(mask, fg_mask)

        if roi_mask is not None:
            mask = cv2.bitwise_and(mask, roi_mask)

        # Rauschen entfernen (Morphologische Operationen)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        
        contours_fill, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for i, cnt in enumerate(contours_fill):
                if hierarchy[0][i][3] != -1: 
                    cv2.drawContours(mask, [cnt], 0, 255, -1)

        contours = self._pick_contours(mask)
        
        if contours:
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                
                if radius < min_radius or radius > max_radius:
                    continue
                    
                perimeter = float(cv2.arcLength(cnt, True))
                circularity = 0.0
                if perimeter > 1e-6:
                    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
                
                score = area * max(0.05, circularity)
                if best is None or score > best[0]:
                    best = (
                        score,
                        Ball2D(center=np.array([x, y], dtype=np.float32), radius=float(radius), confidence=min(1.0, max(0.1, circularity))),
                    )

        if best is not None:
            return best[1], mask

        # Fallback auf Hough Circles
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if roi_mask is not None:
            gray = cv2.bitwise_and(gray, roi_mask)

        blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(10, min_radius * 4),
            param1=max(10, p1),
            param2=max(5, p2),
            minRadius=int(min_radius),
            maxRadius=int(max_radius),
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = max(circles, key=lambda c: c[2])
            return Ball2D(center=np.array([x, y], dtype=np.float32), radius=float(r), confidence=0.3), mask

        return None, mask


class BiomechanicsAnalyzer:
    def __init__(self, fps: float):
        self.dt = 1.0 / max(fps, 1e-6)
        self.kalman = {
            "trunk": Kalman1D(),
            "shoulder": Kalman1D(),
            "elbow": Kalman1D(),
            "knee": Kalman1D()
        }

    def compute_metrics(self, pose3d: Pose3D, pose2d: Pose2D, ball: Optional[Ball2D], state: AnalysisState, current_time: float, pixels_per_meter: Optional[float] = None) -> FrameMetrics:
        kp = pose3d.keypoints
        l_sh = kp.get("left_shoulder")
        r_sh = kp.get("right_shoulder")
        l_hp = kp.get("left_hip")
        r_hp = kp.get("right_hip")
        r_el = kp.get("right_elbow")
        r_wr = kp.get("right_wrist")
        r_kn = kp.get("right_knee")
        r_an = kp.get("right_ankle")
        
        trunk_angle = float('nan')
        shoulder_angle = float('nan')
        elbow_angle = float('nan')
        knee_angle = float('nan')
        speed_mps = 0.0

        if all(x is not None for x in [l_sh, r_sh, l_hp, r_hp]):
            shoulder_center = 0.5 * (l_sh + r_sh)
            hip_center = 0.5 * (l_hp + r_hp)
            torso_vec = shoulder_center - hip_center
            vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            trunk_angle = safe_angle_deg(torso_vec, vertical)
            
            if r_el is not None:
                upper_arm = r_el - r_sh
                shoulder_angle = safe_angle_deg(upper_arm, torso_vec)
                if r_wr is not None:
                    lower_arm = r_wr - r_el
                    elbow_angle = safe_angle_deg(-upper_arm, lower_arm)
                    
        if all(x is not None for x in [r_hp, r_kn, r_an]):
            thigh_vec = r_kn - r_hp
            calf_vec = r_an - r_kn
            knee_angle = safe_angle_deg(-thigh_vec, calf_vec)

        # Primär: Ballgeschwindigkeit messen.
        if ball is not None and pixels_per_meter and pixels_per_meter > 0:
            state.ball_history.append((current_time, ball.center.copy()))
            if len(state.ball_history) > 3:
                state.ball_history.pop(0)
            if len(state.ball_history) >= 2:
                old_time, old_pos = state.ball_history[0]
                new_time, new_pos = state.ball_history[-1]
                dt_actual = new_time - old_time
                if dt_actual > 0.01:
                    pixel_dist = euclidean_distance_3d(old_pos, new_pos)
                    raw_speed = (pixel_dist / pixels_per_meter) / dt_actual
                    if raw_speed < 50.0:
                        speed_mps = raw_speed

        # Fallback: Handgelenkgeschwindigkeit als Schätzer für Ballgeschwindigkeit.
        if speed_mps == 0.0:
            r_wr_2d = pose2d.keypoints.get("right_wrist")
            if r_wr_2d is not None and pixels_per_meter and pixels_per_meter > 0:
                state.wrist_history.append((current_time, r_wr_2d.copy()))
                if len(state.wrist_history) > 3:
                    state.wrist_history.pop(0)
                if len(state.wrist_history) >= 2:
                    old_time, old_pos = state.wrist_history[0]
                    new_time, new_pos = state.wrist_history[-1]
                    dt_actual = new_time - old_time
                    if dt_actual > 0.01:
                        pixel_dist = euclidean_distance_3d(old_pos, new_pos)
                        raw_speed = (pixel_dist / pixels_per_meter) / dt_actual
                        if raw_speed < 42.0:
                            speed_mps = raw_speed * 1.4

        trunk_angle = self.kalman["trunk"].update(trunk_angle)
        shoulder_angle = self.kalman["shoulder"].update(shoulder_angle)
        elbow_angle = self.kalman["elbow"].update(elbow_angle)
        knee_angle = self.kalman["knee"].update(knee_angle)

        return FrameMetrics(current_time, trunk_angle, shoulder_angle, elbow_angle, knee_angle, speed_mps, speed_mps * 3.6)


# --- Analyse- und Reportingfunktionen ---

def calculate_phase_scores(metrics: List[FrameMetrics], target_speed: float) -> Tuple[int, int, int, int]:
    if not metrics or len(metrics) < 5:
        return 0, 0, 0, 0

    valid_metrics = [m for m in metrics if not math.isnan(m.wrist_speed_kmph)]
    if len(valid_metrics) < 5:
        return 0, 0, 0, 0

    peak_idx = max(range(len(valid_metrics)), key=lambda i: valid_metrics[i].wrist_speed_kmph)
    peak_m = valid_metrics[peak_idx]

    start_p1 = max(0, peak_idx - 15)
    end_p1 = max(0, peak_idx - 6)
    phase1 = valid_metrics[start_p1:end_p1]

    start_p2 = max(0, peak_idx - 6)
    phase2 = valid_metrics[start_p2:peak_idx]

    score_p1 = 30.0
    p1_sh = np.mean([m.shoulder_angle_deg for m in phase1 if not math.isnan(m.shoulder_angle_deg)]) if phase1 else float('nan')
    p1_el = np.mean([m.elbow_angle_deg for m in phase1 if not math.isnan(m.elbow_angle_deg)]) if phase1 else float('nan')
    p1_kn = np.mean([m.knee_angle_deg for m in phase1 if not math.isnan(m.knee_angle_deg)]) if phase1 else float('nan')
    
    if not math.isnan(p1_sh):
        if p1_sh < 80: score_p1 -= 5
        elif p1_sh < 90: score_p1 -= 2
        
    if not math.isnan(p1_el):
        if p1_el < 70 or p1_el > 150: score_p1 -= 5
        elif p1_el < 90 or p1_el > 130: score_p1 -= 2
        
    if not math.isnan(p1_kn):
        if p1_kn > 160: score_p1 -= 5 

    score_p2 = 30.0
    if phase2 and len(phase2) > 0:
        p2_tr_start = phase2[0].trunk_inclination_deg
        p2_tr_end = peak_m.trunk_inclination_deg
        if not math.isnan(p2_tr_start) and not math.isnan(p2_tr_end):
            diff = abs(p2_tr_start - p2_tr_end)
            if diff < 5: score_p2 -= 15
            elif diff < 15: score_p2 -= 5
    else:
        score_p2 = 15.0

    max_speed = peak_m.wrist_speed_kmph
    score_p3 = 40.0 * (max_speed / max(1.0, target_speed))
    score_p3 = min(40.0, max(0.0, score_p3))

    total = int(max(0, min(100, score_p1 + score_p2 + score_p3)))
    return int(score_p1), int(score_p2), int(score_p3), total

def build_detailed_report(metrics: List[FrameMetrics], target_speed: float) -> str:
    if not metrics or len(metrics) < 5:
        return "Nicht genügend Messdaten für einen detaillierten Bericht. Bitte spiele den Wurf mindestens einmal durch."

    valid_metrics = [m for m in metrics if not math.isnan(m.wrist_speed_kmph)]
    if len(valid_metrics) < 5:
        return "Keine gültigen Geschwindigkeitsdaten gefunden. Wurde die Kalibrierung durchgeführt?"

    peak_idx = max(range(len(valid_metrics)), key=lambda i: valid_metrics[i].wrist_speed_kmph)
    peak_m = valid_metrics[peak_idx]

    start_p1 = max(0, peak_idx - 15)
    end_p1 = max(0, peak_idx - 6)
    phase1 = valid_metrics[start_p1:end_p1]

    start_p2 = max(0, peak_idx - 6)
    phase2 = valid_metrics[start_p2:peak_idx]
    
    end_p3 = min(len(valid_metrics), peak_idx + 10)
    phase3 = valid_metrics[peak_idx:end_p3]

    p1_sh = np.mean([m.shoulder_angle_deg for m in phase1 if not math.isnan(m.shoulder_angle_deg)]) if phase1 else float('nan')
    p1_el = np.mean([m.elbow_angle_deg for m in phase1 if not math.isnan(m.elbow_angle_deg)]) if phase1 else float('nan')
    p1_kn = np.mean([m.knee_angle_deg for m in phase1 if not math.isnan(m.knee_angle_deg)]) if phase1 else float('nan')
    p1_t_start = phase1[0].time_s if phase1 else peak_m.time_s - 0.3
    p1_t_end = phase1[-1].time_s if phase1 else peak_m.time_s - 0.15

    p2_tr_start = phase2[0].trunk_inclination_deg if phase2 else float('nan')
    p2_tr_end = peak_m.trunk_inclination_deg
    p2_v_start = phase2[0].wrist_speed_kmph if phase2 else 0.0
    p2_v_end = peak_m.wrist_speed_kmph
    p2_t_start = phase2[0].time_s if phase2 else peak_m.time_s - 0.15
    p2_t_end = phase2[-1].time_s if phase2 else peak_m.time_s - 0.03

    p3_tr_max = np.max([m.trunk_inclination_deg for m in phase3 if not math.isnan(m.trunk_inclination_deg)]) if phase3 else peak_m.trunk_inclination_deg

    s_p1, s_p2, s_p3, total_score = calculate_phase_scores(metrics, target_speed)

    report = "1. Gesamt-Wurfscore\n"
    report += f"Dein Score: {total_score} / 100 Punkte\n"
    report += f"  - Ausholphase (Technik & Stand): {s_p1} / 30 Pkt.\n"
    report += f"  - Beschleunigung (Rumpf): {s_p2} / 30 Pkt.\n"
    report += f"  - Abwurf (Power): {s_p3} / 40 Pkt. (Ziel: {target_speed} km/h)\n\n"

    if p1_t_start >= p1_t_end: p1_t_end = p1_t_start + 0.1
    if p2_t_start >= p2_t_end: p2_t_end = p2_t_start + 0.1

    report += "2. Analyse deines Videos im Detail\n"
    report += f"• Ausholphase (ca. {p1_t_start:.2f}s - {p1_t_end:.2f}s):\n"
    if not math.isnan(p1_sh):
        report += f"  Dein Schulterwinkel liegt hier bei ca. {p1_sh:.0f}°. "
        if p1_sh >= 90:
            report += "Das ist hervorragend! Das bedeutet, dein Oberarm ist nicht nur waagerecht, sondern leicht nach oben gerichtet.\n"
        else:
            report += "Der Arm ist etwas tief. Der Oberarm sollte mindestens waagerecht (90°) sein.\n"
    if not math.isnan(p1_el):
        report += f"  Dein Ellbogenwinkel liegt bei ca. {p1_el:.0f}°. "
        if 90 <= p1_el <= 130:
            report += "Der Arm ist also leicht angewinkelt. Du bist hier in einer klassischen, guten Bogenspannung.\n"
        else:
            report += "Achte auf den Hebel. Ideal ist ein leicht angewinkelter Arm (ca. 90-130°).\n"
            
    if not math.isnan(p1_kn):
        report += f"  Dein Kniewinkel auf der Wurfseite beträgt {p1_kn:.0f}°. "
        if p1_kn < 150:
            report += "Du gehst gut in die Knie und nutzt den Beineinsatz (Stemmschritt oder Sprung) für den Wurf!\n"
        else:
            report += "Dein Bein ist fast durchgestreckt. Hier verschenkst du Kraft, die aus den Beinen in den Rumpf fließen könnte.\n"

    report += f"\n• Wurfarm-Beschleunigung (ca. {p2_t_start:.2f}s - {p2_t_end:.2f}s):\n"
    if not math.isnan(p2_tr_start) and not math.isnan(p2_tr_end):
        report += f"  Der Oberkörper (Rumpf) geht dynamisch nach vorne (Winkel sinkt von {p2_tr_start:.0f}° auf {p2_tr_end:.0f}°).\n"
    report += f"  Dein Arm schnellt nach vorne, die Geschwindigkeit steigt von {p2_v_start:.1f} km/h auf {p2_v_end:.1f} km/h.\n"

    report += f"\n• Abwurf & Durchschwung (ca. {peak_m.time_s:.2f}s):\n"
    report += f"  Hier messen wir die Spitzengeschwindigkeit von {p2_v_end:.1f} km/h!\n"
    if not math.isnan(p3_tr_max) and p3_tr_max > p2_tr_end + 5:
        report += f"  Dein Rumpf beugt sich nach dem Abwurf stark nach vorne (bis über {p3_tr_max:.0f}°), was völlig normal ist, da du die gesamte Energie aus dem Oberkörper abbauen musst.\n"

    tips = []
    if not math.isnan(p1_sh) and p1_sh < 80:
        tips.append("- Ellbogen beim Ausholen deutlich höher nehmen! Der Ball verlässt so zu früh die Hand oder die Flugkurve stimmt nicht.")
    if not math.isnan(p1_kn) and p1_kn > 160:
        tips.append("- Arbeite mehr aus den Beinen! Geh im Stemmschritt oder beim Absprung tiefer in die Knie, um Energie aufzubauen.")
    if not math.isnan(p2_tr_start) and not math.isnan(p2_tr_end) and abs(p2_tr_start - p2_tr_end) < 10:
        tips.append("- Rumpf mehr drehen und einsetzen! Du wirfst zu stark aus der reinen Schulterkraft. Tipp: Core-Training.")
    if p2_v_end < target_speed * 0.8:
        tips.append("- Armzug explosiver gestalten. Tipp: Arbeite mit leichten Medizinbällen oder Therabändern an der Schnellkraft.")

    report += "\n3. Fazit & Korrekturen:\n"
    if tips:
        report += "\n".join(tips)
    else:
        report += "Ein sehr dynamischer und schneller Wurf! Die Peitschenbewegung funktioniert extrem gut. Fokus auf Stabilität im Sprunggelenk, um die Kraft sauber zu übertragen."

    return report

def load_reference_sequence(path: Optional[str]) -> List[float]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [float(x) for x in data]
    x = np.linspace(0, 2 * np.pi, 60)
    return list(70 + 35 * np.sin(x))

def select_video_capture(source: str, video_path: Optional[str], camera_index: int) -> Tuple[Any, str]:
    if source == "webcam":
        cap = cv2.VideoCapture(camera_index)
        source_name = f"Webcam[{camera_index}]"
    elif source == "file":
        if not video_path:
            raise RuntimeError("Bitte Video-Datei angeben.")
        cap = cv2.VideoCapture(video_path)
        source_name = video_path
    else:
        raise RuntimeError(f"Unbekannte Quelle: {source}")

    if not cap.isOpened():
        raise RuntimeError(f"Videoquelle konnte nicht geöffnet werden: {source_name}")
    return cap, source_name


# --- Orchestrierung: Die AnalysisEngine ---

class AnalysisEngine:
    def __init__(
        self,
        fps: float,
        reference_seq: List[float],
        pose_backend: str = "auto",
        calibration_mode: str = "body_height",
    ):
        self.fps = fps
        self.pose_estimator = UDPoseEstimator(preferred_backend=pose_backend)
        self.ball_detector = BallDetector()
        self.retargeter = SkeletonRetargeter()
        self.analyzer = BiomechanicsAnalyzer(fps=fps)
        self.per_track_state: Dict[int, AnalysisState] = {1: AnalysisState()}
        self.reference_seq = reference_seq

        self.calibration_mode = "body_height"
        self.pixels_per_meter: Optional[float] = None
        self.calib_points: Optional[List[Tuple[int, int]]] = None
        self.calibration_samples: List[float] = []
        self.calibration_sample_target = 10
        self.calibration_locked = False

        self.ball_cache_by_frame: Dict[int, Ball2D] = {}
        self.set_calibration_mode(calibration_mode, reset=False)

        self.player_bbox: Optional[Tuple[int, int, int, int]] = None

    def set_player_bbox(self, bbox: Optional[Tuple[int, int, int, int]]):
        """Setzt den statischen Bildausschnitt für die Analyse"""
        self.player_bbox = bbox
        self.ball_detector.reset_background_model() # Tracker muss zurückgesetzt werden

    def reset_runtime_state(self) -> None:
        self.analyzer = BiomechanicsAnalyzer(fps=self.fps)
        self.per_track_state = {1: AnalysisState()}

    def clear_ball_cache(self) -> None:
        self.ball_cache_by_frame.clear()

    def set_calibration_mode(self, mode: str, reset: bool = True) -> None:
        if mode not in {"body_height", "manual_line"}:
            mode = "body_height"
        changed = mode != self.calibration_mode
        self.calibration_mode = mode
        if changed and reset:
            self.reset_calibration(keep_mode=True)

    def reset_calibration(self, keep_mode: bool = True) -> None:
        if not keep_mode:
            self.calibration_mode = "body_height"
        self.pixels_per_meter = None
        self.calib_points = None
        self.calibration_samples.clear()
        self.calibration_locked = False

    def get_calibration_progress(self) -> Tuple[int, int, bool]:
        return len(self.calibration_samples), self.calibration_sample_target, self.calibration_locked

    def set_manual_reference(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        reference_length_m: float,
    ) -> bool:
        if reference_length_m <= 0:
            return False

        p1 = np.array(pt1, dtype=np.float32)
        p2 = np.array(pt2, dtype=np.float32)
        pixel_dist = euclidean_distance_3d(p1, p2)
        if pixel_dist < 8.0:
            return False

        self.calibration_mode = "manual_line"
        self.pixels_per_meter = float(pixel_dist / reference_length_m)
        self.calib_points = [(int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1]))]
        self.calibration_samples.clear()
        self.calibration_locked = True
        return True

    def auto_calibrate_from_pose(self, pose2d: Optional[Pose2D], body_height_m: Optional[float]) -> bool:
        if self.calibration_mode != "body_height":
            return False
        if pose2d is None or body_height_m is None or body_height_m <= 0:
            return False

        nose = pose2d.keypoints.get("nose")
        left_ankle = pose2d.keypoints.get("left_ankle")
        right_ankle = pose2d.keypoints.get("right_ankle")
        if nose is None:
            return False

        foot_candidates = [p for p in (left_ankle, right_ankle) if p is not None]
        if not foot_candidates:
            return False

        if len(foot_candidates) == 2:
            foot = 0.5 * (foot_candidates[0] + foot_candidates[1])
        else:
            foot = foot_candidates[0]

        pixel_dist = euclidean_distance_3d(nose, foot)
        if pixel_dist < 30.0:
            return False

        self.calib_points = [(int(nose[0]), int(nose[1])), (int(foot[0]), int(foot[1]))]

        eye_offset_m = min(0.10, max(0.04, body_height_m * 0.04))
        effective_height_m = body_height_m - eye_offset_m
        if effective_height_m <= 0.5:
            return False

        if self.calibration_locked and self.pixels_per_meter:
            return True

        measured_ppm = float(pixel_dist / effective_height_m)
        if not (20.0 <= measured_ppm <= 3000.0):
            return False

        self.calibration_samples.append(measured_ppm)
        if len(self.calibration_samples) > self.calibration_sample_target * 2:
            self.calibration_samples.pop(0)

        self.pixels_per_meter = float(np.median(self.calibration_samples))
        if len(self.calibration_samples) >= self.calibration_sample_target:
            self.calibration_locked = True
        return True

    def set_ball_color_samples(self, bgr_samples: List[Tuple[int, int, int]]) -> bool:
        return self.ball_detector.set_color_filter_from_bgr_samples(bgr_samples)

    def clear_ball_color_filter(self) -> None:
        self.ball_detector.clear_color_filter()

    def has_ball_color_filter(self) -> bool:
        return self.ball_detector.has_color_filter()

    def _store_ball_cache(self, frame_index: int, ball: Ball2D) -> None:
        self.ball_cache_by_frame[int(frame_index)] = Ball2D(
            center=ball.center.copy(),
            radius=float(ball.radius),
            confidence=float(ball.confidence),
        )

    def _get_cached_ball(self, frame_index: Optional[int], frame_shape: Tuple[int, int, int]) -> Tuple[Optional[Ball2D], Optional[np.ndarray]]:
        if frame_index is None:
            return None, None
        cached = self.ball_cache_by_frame.get(int(frame_index))
        if cached is None:
            return None, None

        ball = Ball2D(center=cached.center.copy(), radius=float(cached.radius), confidence=float(cached.confidence))
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cx = int(max(0, min(frame_shape[1] - 1, round(ball.center[0]))))
        cy = int(max(0, min(frame_shape[0] - 1, round(ball.center[1]))))
        cv2.circle(mask, (cx, cy), max(1, int(round(ball.radius))), 255, -1)
        return ball, mask

    def preanalyze_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        kpt_thr: float = 0.3,
        s_max: int = 80,
        v_min: int = 20,
        p1: int = 100,
        p2: int = 20,
        roi_polygon: Optional[List[Tuple[int, int]]] = None,
        use_mog2: bool = True,
        min_rad: float = 15.0,
        max_rad: float = 35.0,
        body_height_m: Optional[float] = None,
    ) -> Tuple[Optional[Pose2D], Optional[Ball2D]]:
        pose2d = self.pose_estimator.estimate(frame, kpt_thr=kpt_thr)
        if self.calibration_mode == "body_height":
            self.auto_calibrate_from_pose(pose2d, body_height_m)

        ball, _ = self.ball_detector.detect(
            frame,
            s_max=s_max,
            v_min=v_min,
            p1=p1,
            p2=p2,
            roi_polygon=roi_polygon,
            use_mog2=use_mog2,
            min_radius=min_rad,
            max_radius=max_rad,
        )
        if ball is not None:
            self._store_ball_cache(frame_index, ball)
        return pose2d, ball

    def init_ball_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        self.ball_detector.init_tracker(frame, bbox)

    def reset_ball_tracker(self):
        self.ball_detector.reset_tracker()

    def process_frame_with_pose(
        self,
        frame: np.ndarray,
        current_time: float,
        kpt_thr: float = 0.3,
        s_max: int = 80,
        v_min: int = 20,
        p1: int = 100,
        p2: int = 20,
        roi_polygon: Optional[List[Tuple[int, int]]] = None,
        target_speed: float = 90.0,
        use_mog2: bool = True,
        min_rad: float = 15.0,
        max_rad: float = 35.0,
        body_height_m: Optional[float] = None,
        frame_index: Optional[int] = None,
    ) -> Tuple[Optional[FrameMetrics], Optional[Pose2D], Optional[Ball2D], Dict[str, np.ndarray], int]:
        
        orig_frame = frame.copy()
        
        # --- NEU: Bild dynamisch zuschneiden, wenn eine Box gezogen wurde ---
        if self.player_bbox is not None and len(self.player_bbox) == 4:
            px, py, pw, ph = self.player_bbox
            # Sicherstellen, dass die Box im Bild bleibt
            px, py = max(0, px), max(0, py)
            pw = min(frame.shape[1] - px, pw)
            ph = min(frame.shape[0] - py, ph)
            
            process_frame = frame[py:py+ph, px:px+pw]
            offset_x, offset_y = px, py
        else:
            process_frame = frame
            offset_x, offset_y = 0, 0

        # Pose & Ball AUF DEM ZUGESCHNITTENEN BILD suchen (bringt den Speed-Boost)
        pose2d = self.pose_estimator.estimate(process_frame, kpt_thr=kpt_thr)

        ball, ball_mask_cropped = None, None
        cached_ball, cached_mask = self._get_cached_ball(frame_index, frame.shape)
        if cached_ball is not None and not self.ball_detector.tracker_active:
            ball, ball_mask = cached_ball, cached_mask
            offset_applied = True # Cache ist immer Full-Frame
        else:
            ball, ball_mask_cropped = self.ball_detector.detect(
                process_frame,
                s_max=s_max, v_min=v_min, p1=p1, p2=p2,
                roi_polygon=roi_polygon, use_mog2=use_mog2,
                min_radius=min_rad, max_radius=max_rad,
            )
            offset_applied = False
            
            # Maske für das Debug-Fenster wieder auf Originalgröße bringen
            ball_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            if ball_mask_cropped is not None:
                bh, bw = ball_mask_cropped.shape[:2]
                ball_mask[py:py+bh, px:px+bw] = ball_mask_cropped

        # Koordinaten wieder auf das Originalbild zurückrechnen!
        if pose2d is not None:
            for k in pose2d.keypoints:
                pose2d.keypoints[k][0] += offset_x
                pose2d.keypoints[k][1] += offset_y
                
        if ball is not None and not offset_applied:
            ball.center[0] += offset_x
            ball.center[1] += offset_y
            if frame_index is not None:
                self._store_ball_cache(frame_index, ball)

        if ball_mask is None:
            ball_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if self.calibration_mode == "body_height":
            self.auto_calibrate_from_pose(pose2d, body_height_m)

        skeleton_frame = np.zeros_like(frame)
        if pose2d is not None:
            draw_pose_overlay(skeleton_frame, pose2d)

        metrics = None
        total_score = 0
        if pose2d is not None:
            pose3d = self.retargeter.retarget(Pose3D(keypoints={k: np.array([v[0], v[1], 0.0]) for k, v in pose2d.keypoints.items()}))
            metrics = self.analyzer.compute_metrics(
                pose3d,
                pose2d,
                ball,
                self.per_track_state[1],
                current_time,
                pixels_per_meter=self.pixels_per_meter,
            )
            self.per_track_state[1].metrics.append(metrics)
            _, _, _, total_score = calculate_phase_scores(self.per_track_state[1].metrics, target_speed)

        final_frame = frame.copy()
        draw_pose_overlay(final_frame, pose2d)
        draw_ball_overlay(final_frame, ball)

        if roi_polygon and len(roi_polygon) >= 3:
            pts = np.array(roi_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(final_frame, [pts], True, (255, 100, 100), 2, cv2.LINE_AA)

        if self.calib_points and len(self.calib_points) == 2:
            pt1, pt2 = self.calib_points
            cv2.circle(final_frame, pt1, 8, (255, 0, 255), 2)
            cv2.circle(final_frame, pt2, 8, (255, 0, 255), 2)
            cv2.line(final_frame, pt1, pt2, (255, 0, 255), 2)

        if metrics:
            if self.pixels_per_meter:
                cv2.putText(final_frame, f"v={metrics.wrist_speed_kmph:.1f} km/h", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(final_frame, f"Score: {total_score} Pkt", (18, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            elif self.calibration_mode == "body_height":
                cv2.putText(final_frame, "v= -- (Auto-Kalibrierung laeuft)", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
            else:
                cv2.putText(final_frame, "v= -- (Referenzlinie setzen)", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        debug_frames = {
            "orig": orig_frame,
            "mask": cv2.cvtColor(ball_mask, cv2.COLOR_GRAY2BGR),
            "skeleton": skeleton_frame,
            "final": final_frame,
        }
        
        return metrics, pose2d, ball, debug_frames, total_score
