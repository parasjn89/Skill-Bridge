"""
Attendance Module (Epic 2)
Handles Offline Participant Face Detection, Persistent Tracking ID Assignment,
Sustained Presence Logic (>70% Session Threshold), and Exportable Attendance Registers.
"""

import os
import math
import numpy as np
import pandas as pd
import cv2

class CentroidTracker:
    """
    Lightweight, fast multi-object centroid tracker for persistent participant identification.
    Assigns stable IDs (STU-01, STU-02...) and tracks presence duration without requiring biometrics.
    """
    def __init__(self, max_disappeared: int = 30, max_distance: float = 85.0):
        self.next_object_id = 1
        self.objects = {}  # id -> centroid (cx, cy)
        self.bboxes = {}   # id -> (x, y, w, h)
        self.disappeared = {}  # id -> count
        self.first_seen = {}   # id -> timestamp (sec)
        self.last_seen = {}    # id -> timestamp (sec)
        self.active_seconds = {}  # id -> accumulated active seconds
        self.head_pose_stats = {} # id -> {'forward': count, 'profile': count, 'slump': count}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, current_time: float, is_profile: bool = False):
        student_id = f"STU-{self.next_object_id:02d}"
        self.objects[student_id] = centroid
        self.bboxes[student_id] = bbox
        self.disappeared[student_id] = 0
        self.first_seen[student_id] = current_time
        self.last_seen[student_id] = current_time
        self.active_seconds[student_id] = 0.0
        self.head_pose_stats[student_id] = {
            'forward': 0 if is_profile else 1,
            'profile': 1 if is_profile else 0,
            'slump': 0
        }
        self.next_object_id += 1
        return student_id

    def deregister(self, object_id):
        # We retain records in first_seen/last_seen/active_seconds for final register compilation,
        # but remove from active spatial tracking
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.bboxes:
            del self.bboxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects, current_time: float, dt: float = 1.0, profile_flags=None):
        """
        Updates participant tracking states with new bounding boxes.
        rects: list of (x, y, w, h)
        """
        if profile_flags is None:
            profile_flags = [False] * len(rects)

        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.bboxes

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2.0), int(y + h / 2.0))

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], rects[i], current_time, profile_flags[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute Euclidean distances between existing and input centroids
            dists = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2
            )
            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if dists[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = rects[col]
                self.disappeared[object_id] = 0
                self.last_seen[object_id] = current_time
                self.active_seconds[object_id] += dt

                # Update pose stats
                is_prof = profile_flags[col]
                if is_prof:
                    self.head_pose_stats[object_id]['profile'] += 1
                else:
                    self.head_pose_stats[object_id]['forward'] += 1

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, dists.shape[0])).difference(used_rows)
            unused_cols = set(range(0, dists.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], rects[col], current_time, profile_flags[col])

        return self.objects, self.bboxes


class ClassroomVisionProcessor:
    """
    Offline video/image processor utilizing OpenCV Haar Cascades for edge detection.
    Guarantees 100% offline edge processing with zero cloud data transmission.
    """
    def __init__(self, models_dir: str = "models"):
        self.frontal_path = os.path.join(models_dir, "haarcascade_frontalface_default.xml")
        self.profile_path = os.path.join(models_dir, "haarcascade_profileface.xml")
        self.eye_path = os.path.join(models_dir, "haarcascade_eye.xml")

        self.face_cascade = cv2.CascadeClassifier(self.frontal_path) if os.path.exists(self.frontal_path) else None
        self.profile_cascade = cv2.CascadeClassifier(self.profile_path) if os.path.exists(self.profile_path) else None
        self.eye_cascade = cv2.CascadeClassifier(self.eye_path) if os.path.exists(self.eye_path) else None

        self.tracker = CentroidTracker(max_disappeared=25, max_distance=90.0)

    def detect_participants(self, frame_bgr: np.ndarray, anonymize: bool = False):
        """
        Detects participants in a single frame.
        Returns:
            processed_frame: BGR frame with visual overlays (and optional blur)
            detections: list of (x, y, w, h)
            profile_flags: list of bools indicating profile (turned away) head pose
            stats: dict of frame level metrics (attentive count, distracted count)
        """
        if frame_bgr is None:
            return None, [], [], {}

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        output_frame = frame_bgr.copy()

        rects = []
        profile_flags = []

        # 1. Frontal faces
        if self.face_cascade and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (x, y, w, h) in faces:
                rects.append((x, y, w, h))
                profile_flags.append(False)

        # 2. Profile faces (head turned sideways)
        if self.profile_cascade and not self.profile_cascade.empty():
            profiles = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30)
            )
            for (x, y, w, h) in profiles:
                # Deduplicate if overlapping with existing frontal face
                overlap = False
                for (fx, fy, fw, fh) in rects:
                    if abs(x - fx) < fw * 0.5 and abs(y - fy) < fh * 0.5:
                        overlap = True
                        break
                if not overlap:
                    rects.append((x, y, w, h))
                    profile_flags.append(True)

        # Optional Anonymization Mode (Ethical Telemetry)
        if anonymize:
            for (x, y, w, h) in rects:
                roi = output_frame[y:y+h, x:x+w]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
                    output_frame[y:y+h, x:x+w] = blurred_roi

        return output_frame, rects, profile_flags

    def process_and_annotate(self, frame_bgr: np.ndarray, current_time: float, dt: float = 1.0, anonymize: bool = False):
        """
        Full detection + tracking + annotation pipeline for a video frame.
        """
        annotated_frame, rects, profile_flags = self.detect_participants(frame_bgr, anonymize=anonymize)
        objects, bboxes = self.tracker.update(rects, current_time, dt, profile_flags)

        attentive_count = 0
        distracted_count = 0

        for student_id, (cx, cy) in objects.items():
            if student_id in bboxes:
                x, y, w, h = bboxes[student_id]
                # Determine pose state
                stats = self.tracker.head_pose_stats.get(student_id, {'forward': 1, 'profile': 0})
                is_attentive = stats['forward'] >= stats['profile']
                if is_attentive:
                    attentive_count += 1
                    box_color = (0, 200, 80) # Emerald Green
                    label_text = f"{student_id} (Attentive)"
                else:
                    distracted_count += 1
                    box_color = (0, 140, 255) # Orange/Amber
                    label_text = f"{student_id} (Turned)"

                # Draw bounding box & persistent ID badge
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 255), -1)

                # Badge label background
                (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame, (x, max(0, y - 20)), (x + lw + 8, y), box_color, -1)
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (x + 4, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        total_visible = len(objects)
        return annotated_frame, total_visible, attentive_count, distracted_count

    def generate_attendance_register(self, session_duration_sec: float, sustained_threshold: float = 0.70):
        """
        Epic 2 Sustained Presence Logic:
        Requires participants to be detected across a sustained duration threshold (>70% session time)
        to prevent false attendance marks from temporary hallway pass-throughs.
        """
        records = []
        all_ids = sorted(list(set(list(self.tracker.first_seen.keys()) + list(self.tracker.active_seconds.keys()))))

        for student_id in all_ids:
            first_s = self.tracker.first_seen.get(student_id, 0.0)
            last_s = self.tracker.last_seen.get(student_id, 0.0)
            dur_s = self.tracker.active_seconds.get(student_id, 0.0)
            
            # Clamp duration to session duration
            effective_dur_s = min(dur_s, session_duration_sec)
            presence_ratio = effective_dur_s / max(session_duration_sec, 1.0)
            presence_pct = round(presence_ratio * 100, 1)

            # Sustained Presence Determination
            if presence_ratio >= sustained_threshold:
                status = "PRESENT"
                notes = "Sustained Classroom Presence Verified"
            elif presence_ratio > 0.0:
                status = "ABSENT"
                notes = f"Transient Pass-Through ({presence_pct}% < {int(sustained_threshold*100)}% threshold)"
            else:
                status = "ABSENT"
                notes = "Not Detected"

            # Format timestamps into MM:SS
            first_fmt = f"{int(first_s // 60):02d}:{int(first_s % 60):02d}"
            last_fmt = f"{int(last_s // 60):02d}:{int(last_s % 60):02d}"
            dur_fmt = f"{int(effective_dur_s // 60)}m {int(effective_dur_s % 60)}s"

            records.append({
                "Student ID": student_id,
                "First Seen": first_fmt,
                "Last Seen": last_fmt,
                "Detected Duration": dur_fmt,
                "Session Presence %": presence_pct,
                "Status": status,
                "Pedagogical Remarks": notes
            })

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                "Student ID", "First Seen", "Last Seen", "Detected Duration", 
                "Session Presence %", "Status", "Pedagogical Remarks"
            ])
        return df
