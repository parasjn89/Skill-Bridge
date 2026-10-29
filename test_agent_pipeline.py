"""
Integration Verification Test for Multimodal Attendance & Engagement Agent
Tests all 5 Epics and core modules end-to-end.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from modules.attendance import ClassroomVisionProcessor, CentroidTracker
from modules.acoustic import AcousticAnalyzer
from modules.fusion import MultimodalFusionEngine
from modules.pedagogical_agent import PedagogicalAgent
from modules.session_analytics import SessionAnalytics
from modules.synthetic_generator import ClassroomSyntheticGenerator

def test_full_pipeline():
    print("[1/6] Testing Models & Vision Processor...")
    vp = ClassroomVisionProcessor(models_dir="models")
    assert vp.face_cascade is not None and not vp.face_cascade.empty(), "Face cascade failed to load!"
    assert vp.profile_cascade is not None and not vp.profile_cascade.empty(), "Profile cascade failed to load!"
    assert vp.eye_cascade is not None and not vp.eye_cascade.empty(), "Eye cascade failed to load!"

    # Test synthetic classroom frame
    student_cfgs = [
        {"id": "STU-01", "pos": (180, 180), "attentive": True, "present": True},
        {"id": "STU-02", "pos": (300, 180), "attentive": False, "present": True},
        {"id": "STU-08", "pos": (650, 240), "attentive": False, "present": True}
    ]
    frame = ClassroomSyntheticGenerator.draw_synthetic_classroom_frame(student_cfgs)
    assert frame is not None and frame.shape == (420, 720, 3)
    annotated, tot, att, dist = vp.process_and_annotate(frame, current_time=10.0, dt=1.0, anonymize=True)
    assert annotated is not None
    print(f" Vision processor passed! Frame shape: {annotated.shape}")

    print("[2/6] Testing Sustained Presence Attendance Logic (>70% threshold)...")
    vis_df, att_df = ClassroomSyntheticGenerator.generate_classroom_session_telemetry(total_seconds=180)
    assert not att_df.empty
    stu1 = att_df[att_df["Student ID"] == "STU-01"].iloc[0]
    stu8 = att_df[att_df["Student ID"] == "STU-08"].iloc[0]
    assert stu1["Status"] == "PRESENT", f"STU-01 should be PRESENT, got {stu1['Status']}"
    assert stu8["Status"] == "ABSENT", f"STU-08 (pass-through) should be ABSENT, got {stu8['Status']}"
    print(f" Sustained presence verified! STU-01 is {stu1['Status']}, Pass-through STU-08 is {stu8['Status']}")

    print("[3/6] Testing Acoustic Dynamics & 4-Class VAD Classification...")
    audio_sig = ClassroomSyntheticGenerator.generate_synthetic_audio(duration_sec=60, sample_rate=16000)
    assert len(audio_sig) == 60 * 16000
    analyzer = AcousticAnalyzer(sample_rate=16000, window_sec=1.0)
    acous_df = analyzer.process_audio_signal(audio_sig, sample_rate=16000)
    assert len(acous_df) == 60
    states_found = set(acous_df["acoustic_state"].unique())
    print(f" Acoustic processor passed! Found acoustic states: {states_found}")

    print("[4/6] Testing Multimodal Fusion Engine (Engagement Index 0-100)...")
    fusion = MultimodalFusionEngine(weight_visual=0.5, weight_audio=0.35, weight_presence=0.15)
    fused_df = fusion.fuse(vis_df, acous_df, total_roster_count=8)
    assert "engagement_index" in fused_df.columns
    assert fused_df["engagement_index"].min() >= 0.0
    assert fused_df["engagement_index"].max() <= 100.0
    print(f" Multimodal fusion passed! Mean Engagement Index: {fused_df['engagement_index'].mean():.1f}%")

    print("[5/6] Testing Heuristic Pedagogical Advisory Agent...")
    ped_agent = PedagogicalAgent(dip_threshold=45.0, dip_duration_window=2)
    suggestions = ped_agent.analyze_session_heuristics(fused_df)
    assert isinstance(suggestions, list)
    print(f" Pedagogical agent passed! Generated {len(suggestions)} real-time intervention prompts.")
    for s in suggestions[:2]:
        print(f"    - [{s['badge']}] {s['message']}")

    print("[6/6] Testing Session Analytics & Report Generation...")
    analytics = SessionAnalytics(fused_df, att_df, suggestions)
    kpis = analytics.compute_kpis()
    assert kpis["attendance_rate"] > 0
    md_rep = analytics.generate_markdown_report()
    html_rep = analytics.generate_html_report()
    assert len(md_rep) > 200
    assert len(html_rep) > 200
    print(f" Session analytics passed! Average Engagement KPI: {kpis['avg_engagement']}%, Attendance: {kpis['attendance_rate']}%")

    print("\n ALL SYSTEM MODULES TESTED & VERIFIED SUCCESFULLY!")

if __name__ == "__main__":
    test_full_pipeline()
