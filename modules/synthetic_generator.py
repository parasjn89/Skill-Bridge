"""
Synthetic Classroom Data & Audio/Video Generator
Provides realistic multi-participant classroom scenarios, synthetic audio signals,
and annotated test frames for 1-click demonstration on Streamlit Cloud.
"""

import numpy as np
import pandas as pd
import cv2

class ClassroomSyntheticGenerator:
    """
    Generates high-fidelity multimodal classroom scenarios for testing Epics 2 through 6.
    """

    @staticmethod
    def generate_synthetic_audio(duration_sec: int = 180, sample_rate: int = 16000) -> np.ndarray:
        """
        Synthesizes a realistic 3-minute multi-phase classroom audio recording:
        - 00:00 - 00:40: Instructor Lecture (steady vocal formants, ~180 Hz pitch + harmonics)
        - 00:40 - 01:20: Student Discussion (active multi-speaker turn-taking, higher dynamic bursts)
        - 01:20 - 01:45: Reflective Silence (room tone, gentle breathing, keyboard clicks)
        - 01:45 - 02:10: High Ambient Classroom Distraction (chair screech, loud dropping/chatter)
        - 02:10 - 03:00: Instructor Re-anchoring & Q&A
        """
        total_samples = duration_sec * sample_rate
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)
        audio = np.random.normal(0, 0.002, total_samples) # Ambient room tone baseline

        # Helper to synthesize human vocal formants
        def make_speech_signal(t_sub, f0=160, formants=(700, 1200, 2400)):
            sig = np.sin(2 * np.pi * f0 * t_sub)
            for f in formants:
                sig += 0.5 * np.sin(2 * np.pi * f * t_sub)
            # Syllabic modulation (4 Hz speech rhythm)
            modulation = (np.sin(2 * np.pi * 4.0 * t_sub) + 1.0) / 2.0
            return sig * modulation * 0.15

        # Phase 1: Instructor Lecture (0 to 45s)
        p1_mask = (t >= 0) & (t < 45)
        audio[p1_mask] += make_speech_signal(t[p1_mask], f0=145, formants=(600, 1100, 2200))

        # Phase 2: Student Discussion (45s to 90s)
        p2_mask = (t >= 45) & (t < 90)
        # Multi-speaker overlap
        spk1 = make_speech_signal(t[p2_mask], f0=190, formants=(750, 1350, 2600))
        spk2 = make_speech_signal(t[p2_mask] * 1.05, f0=220, formants=(800, 1400, 2800))
        audio[p2_mask] += (spk1 * 0.6 + spk2 * 0.5)

        # Phase 3: Reflective Silence (90s to 110s)
        # Only ambient room tone already present

        # Phase 4: High Ambient Distraction (110s to 135s)
        p4_mask = (t >= 110) & (t < 135)
        # Broadband noise + high frequency scraping
        screech = np.sin(2 * np.pi * 3800 * t[p4_mask]) * 0.35
        clatter = np.random.normal(0, 0.15, np.sum(p4_mask))
        audio[p4_mask] += (screech + clatter)

        # Phase 5: Instructor Re-anchoring & Q&A (135s to 180s)
        p5_mask = (t >= 135) & (t < 180)
        audio[p5_mask] += make_speech_signal(t[p5_mask], f0=150, formants=(650, 1150, 2300)) * 0.8

        # Normalize float32
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        return audio.astype(np.float32)

    @staticmethod
    def generate_classroom_session_telemetry(scenario_name: str = "Standard Lecture & Discussion", total_seconds: int = 300) -> tuple:
        """
        Generates realistic synchronized visual telemetry and attendance logs for demonstration.
        Simulates:
        - 8 Roster Students
        - STU-01 through STU-06: Sustained attendance (85% - 98% presence) -> PRESENT
        - STU-07: Late arrival (65% presence) -> ABSENT or border
        - STU-08: Transient hallway pass-through (detected for 18 sec = 6%) -> ABSENT (Pass-through)
        """
        time_steps = list(range(0, total_seconds, 2)) # 2-second resolution
        vis_records = []

        np.random.seed(42)

        for t in time_steps:
            # Pedagogical phase modeling
            if t < 60:
                # Early intro: high attention, 6-7 students visible
                vis_att = np.random.randint(6, 8)
                vis_dist = np.random.randint(0, 2)
            elif 60 <= t < 150:
                # Monologue lecture stretch: engagement gradually dips, 2-3 students slump/look down
                vis_att = np.random.randint(4, 6)
                vis_dist = np.random.randint(2, 4)
            elif 150 <= t < 220:
                # Active Discussion / Poll: engagement rebounds strongly
                vis_att = np.random.randint(6, 8)
                vis_dist = 0
            elif 220 <= t < 250:
                # Distraction event (e.g. noise or door opened)
                vis_att = np.random.randint(2, 4)
                vis_dist = np.random.randint(3, 5)
            else:
                # Wrap-up summary
                vis_att = np.random.randint(5, 7)
                vis_dist = np.random.randint(1, 2)

            # Hallway pass-through student appears briefly between t=80 and t=98
            pass_through = 1 if (80 <= t <= 98) else 0

            tot_visible = vis_att + vis_dist + pass_through

            vis_records.append({
                "time_sec": float(t),
                "total_visible": tot_visible,
                "attentive_count": vis_att,
                "distracted_count": vis_dist + pass_through
            })

        vis_df = pd.DataFrame(vis_records)

        # Pre-seed attendance records with sustained presence logic
        attendance_records = [
            {"Student ID": "STU-01", "First Seen": "00:00", "Last Seen": "05:00", "Detected Duration": "4m 50s", "Session Presence %": 96.7, "Status": "PRESENT", "Pedagogical Remarks": "Sustained Classroom Presence Verified"},
            {"Student ID": "STU-02", "First Seen": "00:00", "Last Seen": "04:58", "Detected Duration": "4m 42s", "Session Presence %": 94.0, "Status": "PRESENT", "Pedagogical Remarks": "Sustained Classroom Presence Verified"},
            {"Student ID": "STU-03", "First Seen": "00:02", "Last Seen": "04:56", "Detected Duration": "4m 30s", "Session Presence %": 90.0, "Status": "PRESENT", "Pedagogical Remarks": "Sustained Classroom Presence Verified"},
            {"Student ID": "STU-04", "First Seen": "00:00", "Last Seen": "04:54", "Detected Duration": "4m 15s", "Session Presence %": 85.0, "Status": "PRESENT", "Pedagogical Remarks": "Sustained Classroom Presence Verified"},
            {"Student ID": "STU-05", "First Seen": "00:04", "Last Seen": "04:50", "Detected Duration": "4m 02s", "Session Presence %": 80.7, "Status": "PRESENT", "Pedagogical Remarks": "Sustained Classroom Presence Verified"},
            {"Student ID": "STU-06", "First Seen": "00:00", "Last Seen": "04:46", "Detected Duration": "3m 48s", "Session Presence %": 76.0, "Status": "PRESENT", "Pedagogical Remarks": "Sustained Classroom Presence Verified"},
            {"Student ID": "STU-07", "First Seen": "01:45", "Last Seen": "04:50", "Detected Duration": "3m 05s", "Session Presence %": 61.7, "Status": "ABSENT", "Pedagogical Remarks": "Transient / Late Presence (61.7% < 70% threshold)"},
            {"Student ID": "STU-08", "First Seen": "01:20", "Last Seen": "01:38", "Detected Duration": "0m 18s", "Session Presence %": 6.0, "Status": "ABSENT", "Pedagogical Remarks": "Transient Pass-Through (6.0% < 70% threshold)"},
        ]
        att_df = pd.DataFrame(attendance_records)

        return vis_df, att_df

    @staticmethod
    def draw_synthetic_classroom_frame(student_configs: list, frame_w: int = 720, frame_h: int = 420) -> np.ndarray:
        """
        Draws a realistic stylized classroom video frame with students seated at desks,
        suitable for live rendering in the UI.
        """
        # Background: Warm modern classroom aesthetic
        frame = np.full((frame_h, frame_w, 3), (35, 30, 28), dtype=np.uint8)

        # Draw Classroom Whiteboard / Screen
        cv2.rectangle(frame, (120, 20), (600, 110), (50, 45, 40), -1)
        cv2.rectangle(frame, (120, 20), (600, 110), (120, 110, 100), 2)
        cv2.putText(frame, "LECTURE: MACHINE LEARNING & COGNITIVE VISION", (145, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 220), 1, cv2.LINE_AA)

        # Draw Student Desks and Avatars
        for cfg in student_configs:
            sx, sy = cfg["pos"]
            stu_id = cfg["id"]
            is_attentive = cfg.get("attentive", True)
            is_present = cfg.get("present", True)

            if not is_present:
                continue

            # Desk
            cv2.rectangle(frame, (sx - 35, sy + 30), (sx + 35, sy + 50), (60, 50, 45), -1)
            cv2.rectangle(frame, (sx - 35, sy + 30), (sx + 35, sy + 50), (80, 70, 65), 1)

            # Torso / Body
            body_color = cfg.get("shirt_color", (130, 80, 50))
            cv2.ellipse(frame, (sx, sy + 32), (24, 18), 0, 0, 360, body_color, -1)

            # Head Oval
            head_color = (195, 205, 230) # Skin tone
            cv2.ellipse(frame, (sx, sy), (15, 20), 0, 0, 360, head_color, -1)

            # Hair
            cv2.ellipse(frame, (sx, sy - 8), (16, 12), 0, 180, 360, (20, 20, 30), -1)

            # Eyes & Gaze Direction
            if is_attentive:
                # Forward looking eyes
                cv2.circle(frame, (sx - 5, sy - 2), 2, (30, 30, 40), -1)
                cv2.circle(frame, (sx + 5, sy - 2), 2, (30, 30, 40), -1)
                box_color = (40, 200, 80)
                status_lbl = f"{stu_id} (Attentive)"
            else:
                # Turned head / looking sideways or down
                cv2.circle(frame, (sx + 7, sy - 1), 2, (30, 30, 40), -1)
                box_color = (30, 130, 240)
                status_lbl = f"{stu_id} (Turned)"

            # Bounding Box overlay
            bw, bh = 54, 76
            bx, by = sx - 27, sy - 25
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_color, 2)
            cv2.rectangle(frame, (bx, by - 16), (bx + bw, by), box_color, -1)
            cv2.putText(frame, status_lbl, (bx + 2, by - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
