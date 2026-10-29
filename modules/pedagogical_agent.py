"""
AI Teaching Suggestion Engine (Epic 5)
Heuristic Advisory Agent generating real-time, non-intrusive pedagogical prompts
when attention flags, monologue fatigue occurs, or distraction spikes.
"""

import pandas as pd
import numpy as np

class PedagogicalAgent:
    """
    Expert Pedagogical Intelligence & Advisory Heuristic Agent.
    Evaluates rolling classroom dynamics and provides empathetic, constructive teaching interventions.
    """
    def __init__(self, dip_threshold: float = 45.0, dip_duration_window: int = 3):
        self.dip_threshold = dip_threshold
        self.dip_duration_window = dip_duration_window

    def analyze_session_heuristics(self, fused_df: pd.DataFrame) -> list:
        """
        Inspects the multimodal time-step matrix and generates prioritized pedagogical alerts.
        """
        suggestions = []
        if fused_df.empty or len(fused_df) < 5:
            return suggestions

        total_windows = len(fused_df)
        window_duration_sec = fused_df["time_sec"].iloc[1] - fused_df["time_sec"].iloc[0] if total_windows > 1 else 1.0

        # Heuristic 1: Collective Engagement Dips (<45% for sustained duration)
        consecutive_low = 0
        start_low_idx = 0
        for i, row in fused_df.iterrows():
            if row["engagement_index"] < self.dip_threshold:
                if consecutive_low == 0:
                    start_low_idx = i
                consecutive_low += 1
            else:
                if consecutive_low >= self.dip_duration_window:
                    dur_min = round((consecutive_low * window_duration_sec) / 60.0, 1)
                    time_lbl = fused_df.loc[start_low_idx, "time_label"]
                    avg_low_ei = fused_df.loc[start_low_idx:i-1, "engagement_index"].mean()
                    suggestions.append({
                        "type": "urgent",
                        "badge": "⚠️ ATTENTION DIP ALERT",
                        "timestamp": time_lbl,
                        "time_sec": fused_df.loc[start_low_idx, "time_sec"],
                        "message": (
                            f"Collective engagement dropped to {avg_low_ei:.1f}% (<{self.dip_threshold}%) for "
                            f"{dur_min} minutes (at {time_lbl})."
                        ),
                        "recommendation": (
                            "Recommend launching a quick 60-second Poll, an interactive Think-Pair-Share, "
                            "or a 2-minute physical stretch break to reset classroom cognitive bandwidth."
                        ),
                        "action_category": "Interactive Reset"
                    })
                consecutive_low = 0

        # Catch trailing dip
        if consecutive_low >= self.dip_duration_window:
            dur_min = round((consecutive_low * window_duration_sec) / 60.0, 1)
            time_lbl = fused_df.loc[start_low_idx, "time_label"]
            avg_low_ei = fused_df.loc[start_low_idx:, "engagement_index"].mean()
            suggestions.append({
                "type": "urgent",
                "badge": "⚠️ ATTENTION DIP ALERT",
                "timestamp": time_lbl,
                "time_sec": fused_df.loc[start_low_idx, "time_sec"],
                "message": (
                    f"Collective engagement dropped to {avg_low_ei:.1f}% for the past {dur_min} minutes."
                ),
                "recommendation": (
                    "Attention has flagged. Launch a quick formative poll or solicit questions from students."
                ),
                "action_category": "Interactive Reset"
            })

        # Heuristic 2: Monologue Fatigue (Continuous Lecture > 15 mins or > 30% of session without verbal exchange)
        lecture_streak = 0
        lecture_start = 0
        lecture_threshold_sec = 180 # 3 min for demo or 900 for full lecture
        for i, row in fused_df.iterrows():
            if row.get("acoustic_state") == "Instructor Lecture":
                if lecture_streak == 0:
                    lecture_start = i
                lecture_streak += 1
            else:
                dur_streak_sec = lecture_streak * window_duration_sec
                if dur_streak_sec >= lecture_threshold_sec:
                    t_min = round(dur_streak_sec / 60.0, 1)
                    t_lbl = fused_df.loc[lecture_start, "time_label"]
                    suggestions.append({
                        "type": "advisory",
                        "badge": "🗣️ MONOLOGUE FATIGUE",
                        "timestamp": t_lbl,
                        "time_sec": fused_df.loc[lecture_start, "time_sec"],
                        "message": (
                            f"Unbroken instructor lecture detected for {t_min} continuous minutes ({t_lbl}) "
                            f"with 0% student verbal interaction."
                        ),
                        "recommendation": (
                            "Break the monologue: pose an open-ended concept check question, "
                            "invite student predictions, or prompt peer discussion."
                        ),
                        "action_category": "Active Inquiry"
                    })
                lecture_streak = 0

        # Heuristic 3: High Ambient Classroom Distraction Spikes
        distraction_df = fused_df[fused_df.get("acoustic_state", "") == "High Ambient Classroom Distraction"]
        if not distraction_df.empty:
            grouped = (distraction_df["time_sec"].diff() > (window_duration_sec * 2)).cumsum()
            for _, group in distraction_df.groupby(grouped):
                if len(group) >= 2:
                    t_lbl = group["time_label"].iloc[0]
                    avg_db = group.get("dbfs", pd.Series([-20.0])).mean()
                    suggestions.append({
                        "type": "advisory",
                        "badge": "🔊 DISTRACTION OVERLOAD",
                        "timestamp": t_lbl,
                        "time_sec": group["time_sec"].iloc[0],
                        "message": (
                            f"Elevated acoustic classroom disturbance recorded ({avg_db:.1f} dBFS at {t_lbl})."
                        ),
                        "recommendation": (
                            "Re-anchor collective attention: Use a clear auditory verbal cue: "
                            "'Let's bring our focus back to the board before we proceed'."
                        ),
                        "action_category": "Classroom Management"
                    })

        # Heuristic 4: Posture Slump / Physical Lethargy
        if "visual_attention_score" in fused_df.columns:
            slump_df = fused_df[fused_df["visual_attention_score"] < 40.0]
            if len(slump_df) >= 3:
                t_lbl = slump_df["time_label"].iloc[0]
                suggestions.append({
                    "type": "advisory",
                    "badge": "💤 POSTURE SLUMP / LETHARGY",
                    "timestamp": t_lbl,
                    "time_sec": slump_df["time_sec"].iloc[0],
                    "message": (
                        f"Over 60% of visible students showed downward gaze or non-forward head posture at {t_lbl}."
                    ),
                    "recommendation": (
                        "Physical energy is low. Guide a quick stand-up stretch, "
                        "switch from passive slides to live board demonstration, or ask students to write for 1 minute."
                    ),
                    "action_category": "Physical Energy Reset"
                })

        # Heuristic 5: Flow State / Engagement Peak Momentum (Positive Reinforcement)
        flow_df = fused_df[fused_df["engagement_index"] >= 80.0]
        if len(flow_df) >= 3:
            t_lbl = flow_df["time_label"].iloc[0]
            suggestions.append({
                "type": "praise",
                "badge": "🌟 HIGH ENGAGEMENT FLOW",
                "timestamp": t_lbl,
                "time_sec": flow_df["time_sec"].iloc[0],
                "message": (
                    f"Exceptional collective engagement peak ({flow_df['engagement_index'].max():.1f}%) "
                    f"achieved at {t_lbl} during collaborative dialogue."
                ),
                "recommendation": (
                    "High momentum detected! Ideal moment to solidify the core takeaway or assign a challenge extension."
                ),
                "action_category": "Concept Reinforcement"
            })

        # Deduplicate suggestions by timestamp & type
        unique_suggestions = []
        seen = set()
        for s in suggestions:
            key = (s["badge"], s["timestamp"])
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)

        # Sort chronologically
        unique_suggestions.sort(key=lambda x: x["time_sec"])
        return unique_suggestions
