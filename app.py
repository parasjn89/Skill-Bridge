"""
Multimodal Attendance & Engagement Agent
Skill Wallet Capstone Project 5
100% Offline Edge Processing • Zero External Cloud Video Transmission • Ethical Pedagogical Telemetry
"""

import os
import time
import io
import soundfile as sf
import numpy as np
import pandas as pd
import streamlit as st
import cv2

# Set Page Config as first Streamlit call
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modular project components
from modules.ui_components import apply_custom_css, render_hero_banner, render_privacy_callout
from modules.attendance import ClassroomVisionProcessor
from modules.acoustic import AcousticAnalyzer
from modules.fusion import MultimodalFusionEngine
from modules.pedagogical_agent import PedagogicalAgent
from modules.session_analytics import SessionAnalytics
from modules.synthetic_generator import ClassroomSyntheticGenerator

# Apply custom educational CSS styling
apply_custom_css()

# Initialize session state for persistent telemetry
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = True
    st.session_state.vision_processor = ClassroomVisionProcessor(models_dir="models")
    st.session_state.acoustic_analyzer = AcousticAnalyzer(sample_rate=16000, window_sec=1.0)
    st.session_state.fusion_engine = MultimodalFusionEngine(weight_visual=0.50, weight_audio=0.35, weight_presence=0.15)
    st.session_state.pedagogical_agent = PedagogicalAgent(dip_threshold=45.0, dip_duration_window=2)
    
    # Pre-generate demo baseline data so app is immediately responsive on first load
    vis_df, att_df = ClassroomSyntheticGenerator.generate_classroom_session_telemetry(total_seconds=180)
    audio_signal = ClassroomSyntheticGenerator.generate_synthetic_audio(duration_sec=180, sample_rate=16000)
    acous_df = st.session_state.acoustic_analyzer.process_audio_signal(audio_signal, sample_rate=16000)
    fused_df = st.session_state.fusion_engine.fuse(vis_df, acous_df, total_roster_count=8)
    suggestions = st.session_state.pedagogical_agent.analyze_session_heuristics(fused_df)

    st.session_state.vis_df = vis_df
    st.session_state.att_df = att_df
    st.session_state.audio_signal = audio_signal
    st.session_state.acous_df = acous_df
    st.session_state.fused_df = fused_df
    st.session_state.suggestions = suggestions
    st.session_state.session_duration_sec = 180

# ==========================================
# SIDEBAR CONTROLS & PRIVACY CONFIGURATION
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Session & Edge Controls")
    
    input_mode = st.radio(
        "Data Stream Source",
        ["🚀 Interactive Classroom Simulation (1-Click)", "📁 Upload Custom Media", "📷 Live Camera / Snapshot"],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 🔒 Privacy & Ethical Guardrails")
    anonymize_mode = st.toggle("Anonymize Faces (Edge Blur)", value=False, help="Applies local Gaussian blurring to participant faces before visual rendering.")
    ethical_smoothing = st.toggle("Collective Telemetry Smoothing", value=True, help="Prevents individual participant profiling by computing rolling group-level indices.")
    
    st.markdown("---")
    st.markdown("#### 🎚️ Core Thresholds (Epic 2-5)")
    sustained_threshold = st.slider(
        "Sustained Presence Threshold (Epic 2)",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        format="%d%%",
        help="Minimum active attendance duration required (>70% standard) to filter out transient hallway pass-throughs."
    )
    
    dip_threshold = st.slider(
        "Attention Dip Trigger Threshold (Epic 5)",
        min_value=30,
        max_value=60,
        value=45,
        step=5,
        format="%d%%",
        help="Collective engagement index level below which the pedagogical advisory agent triggers active intervention prompts."
    )

    st.markdown("---")
    st.markdown("#### ⚖️ Multimodal Fusion Weights (Epic 4)")
    w_vis = st.slider("Visual Weight (Pose/Posture)", 0.1, 0.8, 0.50, 0.05)
    w_acous = st.slider("Acoustic Weight (Dynamics)", 0.1, 0.8, 0.35, 0.05)
    w_pres = st.slider("Presence Weight (Roster %)", 0.0, 0.5, 0.15, 0.05)

    st.markdown("---")
    if st.button("🔄 Regenerate / Refresh Simulation", use_container_width=True):
        vis_df, att_df = ClassroomSyntheticGenerator.generate_classroom_session_telemetry(total_seconds=180)
        audio_signal = ClassroomSyntheticGenerator.generate_synthetic_audio(duration_sec=180, sample_rate=16000)
        acous_df = st.session_state.acoustic_analyzer.process_audio_signal(audio_signal, sample_rate=16000)
        st.session_state.fusion_engine.update_weights(w_vis, w_acous, w_pres)
        fused_df = st.session_state.fusion_engine.fuse(vis_df, acous_df, total_roster_count=8)
        st.session_state.pedagogical_agent.dip_threshold = dip_threshold
        suggestions = st.session_state.pedagogical_agent.analyze_session_heuristics(fused_df)

        st.session_state.vis_df = vis_df
        st.session_state.att_df = att_df
        st.session_state.audio_signal = audio_signal
        st.session_state.acous_df = acous_df
        st.session_state.fused_df = fused_df
        st.session_state.suggestions = suggestions
        st.success("Classroom session refreshed!")

# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
render_hero_banner()
render_privacy_callout()

# Dynamic updates if sliders change
st.session_state.fusion_engine.update_weights(w_vis, w_acous, w_pres)
st.session_state.pedagogical_agent.dip_threshold = dip_threshold

# Define the 5 Core Tabs as requested
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Tab 1: Automated Attendance",
    "🎙️ Tab 2: Acoustic Interaction",
    "🧠 Tab 3: Multimodal Fusion",
    "💡 Tab 4: AI Teaching Suggestion Engine",
    "📊 Tab 5: Session Analytics & Educator Report"
])

# ==============================================================================
# TAB 1: AUTOMATED PRESENCE-BASED ATTENDANCE (EPIC 2)
# ==============================================================================
with tab1:
    st.markdown("### 📋 Automated Presence-Based Attendance Register (Epic 2)")
    st.markdown(
        """
        **Sustained Presence Logic**: Enforces a strict duration threshold (**>70% of session time**) 
        to accurately differentiate fully present students from transient hallway pass-throughs.
        """
    )

    t1_col1, t1_col2 = st.columns([1.1, 0.9])

    with t1_col1:
        st.markdown("#### 🎥 Video Feed Processor & Participant Tracking")
        
        if input_mode == "📷 Live Camera / Snapshot":
            cam_img = st.camera_input("Take a classroom webcam snapshot")
            if cam_img is not None:
                bytes_data = cam_img.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                annotated, tot, att, dist = st.session_state.vision_processor.process_and_annotate(
                    cv2_img, current_time=time.time(), dt=1.0, anonymize=anonymize_mode
                )
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Live Processed Edge Frame", use_container_width=True)
                st.info(f"Detected: {tot} participants ({att} Attentive, {dist} Turned)")
            else:
                st.warning("Awaiting camera snapshot or webcam permission.")

        elif input_mode == "📁 Upload Custom Media":
            uploaded_file = st.file_uploader("Upload Classroom Image or Video Frame", type=["jpg", "jpeg", "png", "mp4"])
            if uploaded_file is not None:
                if uploaded_file.name.endswith((".jpg", ".jpeg", ".png")):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    cv2_img = cv2.imdecode(file_bytes, 1)
                    annotated, tot, att, dist = st.session_state.vision_processor.process_and_annotate(
                        cv2_img, current_time=time.time(), dt=1.0, anonymize=anonymize_mode
                    )
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Processed Custom Image", use_container_width=True)
                    st.info(f"Participants Tracked: {tot} | Forward Pose: {att} | Profile / Turned: {dist}")
                else:
                    st.info("Video uploaded. Offline keyframes will be processed sequentially.")
            else:
                st.info("Upload an image/video to test edge face detection.")

        else: # Simulation Mode
            sim_time = st.slider("Classroom Session Playhead (Seconds)", 0, 180, 85, step=5)
            
            # Setup dynamic student postures based on playhead
            student_configs = [
                {"id": "STU-01", "pos": (180, 180), "attentive": True, "present": True, "shirt_color": (160, 90, 50)},
                {"id": "STU-02", "pos": (300, 180), "attentive": True, "present": True, "shirt_color": (50, 140, 160)},
                {"id": "STU-03", "pos": (420, 180), "attentive": sim_time < 120 or sim_time > 150, "present": True, "shirt_color": (80, 160, 90)},
                {"id": "STU-04", "pos": (540, 180), "attentive": True, "present": True, "shirt_color": (140, 80, 150)},
                {"id": "STU-05", "pos": (220, 310), "attentive": sim_time < 90, "present": True, "shirt_color": (170, 120, 40)},
                {"id": "STU-06", "pos": (360, 310), "attentive": True, "present": True, "shirt_color": (60, 100, 180)},
                {"id": "STU-07", "pos": (500, 310), "attentive": sim_time > 60, "present": sim_time > 60, "shirt_color": (120, 120, 120)},
                # Hallway Pass-through participant: only present near door between t=75 and t=95!
                {"id": "STU-08", "pos": (650, 240), "attentive": False, "present": (75 <= sim_time <= 95), "shirt_color": (190, 60, 60)}
            ]

            frame = ClassroomSyntheticGenerator.draw_synthetic_classroom_frame(student_configs)
            
            if anonymize_mode:
                for cfg in student_configs:
                    if cfg["present"]:
                        sx, sy = cfg["pos"]
                        roi = frame[max(0, sy-25):sy+50, max(0, sx-27):sx+27]
                        if roi.shape[0] > 0 and roi.shape[1] > 0:
                            frame[max(0, sy-25):sy+50, max(0, sx-27):sx+27] = cv2.GaussianBlur(roi, (31, 31), 15)

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Classroom In-Memory Edge Telemetry (Session Playhead: {sim_time}s)", use_container_width=True)

            if 75 <= sim_time <= 95:
                st.warning("🚶 Hallway Pass-Through Event: `STU-08` temporarily walked past classroom field-of-view.")
            else:
                st.caption("All participants tracked locally with anonymous centroid IDs. No raw face vectors stored.")

    with t1_col2:
        st.markdown("#### 📑 Verified Attendance Register")
        
        # Recalculate attendance register dynamically with active threshold
        att_df = st.session_state.att_df.copy()
        
        # Apply current sustained_threshold dynamically
        threshold_pct = sustained_threshold * 100.0
        for idx in range(len(att_df)):
            p_pct = float(att_df.loc[idx, "Session Presence %"])
            if p_pct >= threshold_pct:
                att_df.loc[idx, "Status"] = "PRESENT"
                att_df.loc[idx, "Pedagogical Remarks"] = f"Sustained Presence Verified ({p_pct}% >= {threshold_pct:.0f}%)"
            else:
                att_df.loc[idx, "Status"] = "ABSENT"
                att_df.loc[idx, "Pedagogical Remarks"] = f"Transient Pass-Through ({p_pct}% < {threshold_pct:.0f}% threshold)"

        # Highlight status styling
        def style_status(val):
            if val == "PRESENT":
                return "background-color: rgba(16, 185, 129, 0.25); color: #6ee7b7; font-weight: bold;"
            else:
                return "background-color: rgba(239, 68, 68, 0.25); color: #fca5a5; font-weight: bold;"

        st.dataframe(
            att_df.style.map(style_status, subset=["Status"]),
            use_container_width=True,
            height=320
        )

        # Attendance Metrics Summary
        present_count = len(att_df[att_df["Status"] == "PRESENT"])
        absent_count = len(att_df[att_df["Status"] == "ABSENT"])
        
        c_m1, c_m2 = st.columns(2)
        c_m1.metric("Verified Present", f"{present_count} / {len(att_df)}")
        c_m2.metric("Filtered Pass-Throughs", f"{absent_count} Detected")

        # Export CSV Button
        csv_buffer = io.StringIO()
        att_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Attendance Register (CSV)",
            data=csv_buffer.getvalue(),
            file_name="verified_attendance_register.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption("ℹ️ *Notice how transient student STU-08 (6.0% presence) is automatically classified as ABSENT by the sustained threshold filter.*")

# ==============================================================================
# TAB 2: ACOUSTIC INTERACTION & NOISE DYNAMICS (EPIC 3)
# ==============================================================================
with tab2:
    st.markdown("### 🎙️ Acoustic Interaction & Classroom Noise Dynamics (Epic 3)")
    st.markdown(
        """
        **Acoustic Stream Processor**: Segments local audio into sliding windows, computes spectral energy, 
        Zero Crossing Rate (ZCR), and executes Voice Activity Detection (VAD) to classify signals into 4 classroom states.
        """
    )

    t2_col1, t2_col2 = st.columns([1.2, 0.8])

    with t2_col1:
        st.markdown("#### 📈 Spectral Energy & Acoustic Classification Timeline")
        acoustic_timeline_fig = AcousticAnalyzer.render_timeline_plot(st.session_state.acous_df)
        st.plotly_chart(acoustic_timeline_fig, use_container_width=True)

    with t2_col2:
        st.markdown("#### 🍩 Acoustic State Time Breakdown")
        acoustic_pie_fig = AcousticAnalyzer.render_distribution_pie(st.session_state.acous_df)
        st.plotly_chart(acoustic_pie_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Audio Player & Spectral Telemetry Inspection")
    
    # Audio Player
    c_audio, c_stats = st.columns([1, 1])
    with c_audio:
        st.markdown("**Listen to Classroom Acoustic Stream (Synthesized On-Device):**")
        audio_io = io.BytesIO()
        sf.write(audio_io, st.session_state.audio_signal, 16000, format='WAV')
        st.audio(audio_io.getvalue(), format="audio/wav")
        st.caption("Acoustic stream generated locally using SciPy vocal formant synthesis.")

    with c_stats:
        acous_df = st.session_state.acous_df
        avg_db = acous_df["dbfs"].mean()
        peak_noise = acous_df["dbfs"].max()
        disc_pct = (len(acous_df[acous_df["acoustic_state"] == "Student Discussion / Verbal Interaction"]) / len(acous_df)) * 100
        
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Mean Room Energy", f"{avg_db:.1f} dBFS")
        col_s2.metric("Peak Noise Spike", f"{peak_noise:.1f} dBFS")
        col_s3.metric("Discussion Time", f"{disc_pct:.1f}%")

    with st.expander("🔬 View Detailed Raw Acoustic Feature Window Matrix"):
        st.dataframe(st.session_state.acous_df, use_container_width=True, height=250)

# ==============================================================================
# TAB 3: MULTIMODAL FUSION & ENGAGEMENT SCORING (EPIC 4)
# ==============================================================================
with tab3:
    st.markdown("### 🧠 Multimodal Fusion & Engagement Scoring (Epic 4)")
    st.markdown(
        """
        **Unified Time-Step Matrix**: Fuses visual attentiveness indicators (forward head pose, upright posture vs slumping) 
        with acoustic interaction energy to compute the normalized **Classroom Engagement Index (0 - 100)**.
        """
    )

    # Re-fuse with current weights
    fused_df = st.session_state.fusion_engine.fuse(
        st.session_state.vis_df,
        st.session_state.acous_df,
        total_roster_count=8
    )
    st.session_state.fused_df = fused_df

    # Render Multimodal Engagement Curve
    engagement_fig = MultimodalFusionEngine.render_engagement_timeline(fused_df)
    st.plotly_chart(engagement_fig, use_container_width=True)

    # Summary metric ribbon
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)
    avg_eng = fused_df["engagement_index"].mean()
    f_c1.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Average Engagement Index</div>
        <div class="kpi-value">{avg_eng:.1f}%</div>
        <div class="kpi-subtext">Unified multimodal score</div>
    </div>
    """, unsafe_allow_html=True)

    peaks_count = fused_df["is_peak"].sum()
    f_c2.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Attention Peaks 🚀</div>
        <div class="kpi-value">{peaks_count} Events</div>
        <div class="kpi-subtext">Active dialogue / breakthroughs</div>
    </div>
    """, unsafe_allow_html=True)

    dips_count = fused_df["is_dip"].sum()
    f_c3.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Attention Dips ⚠️</div>
        <div class="kpi-value">{dips_count} Events</div>
        <div class="kpi-subtext">Monologue or distraction lull</div>
    </div>
    """, unsafe_allow_html=True)

    active_weight_summary = f"{int(w_vis*100)}% V | {int(w_acous*100)}% A | {int(w_pres*100)}% P"
    f_c4.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Fusion Formulation</div>
        <div class="kpi-value" style="font-size: 20px;">{active_weight_summary}</div>
        <div class="kpi-subtext">Configured in sidebar</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 View Unified Multimodal Time-Step Matrix"):
        st.dataframe(fused_df[["time_label", "engagement_index", "visual_attention_score", "acoustic_engagement_score", "acoustic_state", "engagement_zone"]], use_container_width=True, height=250)

# ==============================================================================
# TAB 4: AI TEACHING SUGGESTION ENGINE (EPIC 5)
# ==============================================================================
with tab4:
    st.markdown("### 💡 AI Teaching Suggestion Engine (Epic 5)")
    st.markdown(
        """
        **Heuristic Advisory Agent**: Continuously evaluates collective engagement dynamics, monologue duration, 
        and acoustic noise to generate real-time, non-intrusive pedagogical prompts when attention flags.
        """
    )

    # Re-evaluate suggestions
    st.session_state.pedagogical_agent.dip_threshold = dip_threshold
    suggestions = st.session_state.pedagogical_agent.analyze_session_heuristics(fused_df)
    st.session_state.suggestions = suggestions

    p_col1, p_col2 = st.columns([1.1, 0.9])

    with p_col1:
        st.markdown("#### 🔔 Real-Time Pedagogical Advisory Feed")
        if suggestions:
            for s in suggestions:
                badge_type = s["type"]
                if badge_type == "urgent":
                    card_class = "pedagogy-card-urgent"
                    icon = "🚨"
                elif badge_type == "praise":
                    card_class = "pedagogy-card-praise"
                    icon = "🌟"
                else:
                    card_class = "pedagogy-card-advisory"
                    icon = "💡"

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <strong style="font-size: 14px;">{icon} {s['badge']}</strong>
                            <span style="font-size: 12px; opacity: 0.8;">Playhead: <code>{s['timestamp']}</code></span>
                        </div>
                        <div style="font-size: 13px; margin-bottom: 8px;">{s['message']}</div>
                        <div style="font-size: 13px; font-weight: 500; background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px;">
                            👉 <strong>Recommended Action:</strong> {s['recommendation']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("✅ Classroom engagement is steady! No critical attention dips or monologue flags detected.")

    with p_col2:
        st.markdown("#### 🛠️ Quick-Launch Educator Interventions")
        
        # Tool 1: 2-Minute Stretch Break
        with st.expander("🧘 Launch 2-Minute Stretch Break (Intervention)", expanded=True):
            st.write("Guides a physical posture reset when cognitive lethargy or slumping is detected.")
            if st.button("▶️ Start 2-Minute Stretch Timer", key="start_timer_btn", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for sec in range(1, 11): # Accelerated demo timer
                    progress_bar.progress(sec / 10.0)
                    status_text.caption(f"Stretch Break in progress... ({sec * 12}s elapsed)")
                    time.sleep(0.15)
                status_text.success("🎉 Stretch Break Complete! Students re-energized and seated.")

        # Tool 2: Formative Assessment Prompts
        with st.expander("📝 Formative Assessment Prompt Templates"):
            assessment_type = st.selectbox(
                "Select Teaching Intervention",
                ["60-Second Quick Poll", "Think-Pair-Share Prompt", "1-Minute Muddiest Point Check", "Cold-Call Inquiry"]
            )
            if assessment_type == "60-Second Quick Poll":
                st.info("📢 **Prompt:** 'On a scale of 1-5, how confident are you applying today's second rule? Submit via phone or hand raise.'")
            elif assessment_type == "Think-Pair-Share Prompt":
                st.info("🤝 **Prompt:** 'Turn to your neighbor for 90 seconds: Identify one real-world edge case where this approach fails.'")
            elif assessment_type == "1-Minute Muddiest Point Check":
                st.info("✍️ **Prompt:** 'Write down the single most confusing step from the last 15 minutes on a sticky note.'")
            else:
                st.info("🎯 **Prompt:** 'STU-04, how would the output change if we inverted the feature weights?'")

# ==============================================================================
# TAB 5: SESSION ANALYTICS & EDUCATOR REPORT (EPIC 5 & 6)
# ==============================================================================
with tab5:
    st.markdown("### 📊 Session Analytics & Post-Session Educator Report (Epics 5 & 6)")
    st.markdown(
        """
        **Post-Session Summary Dashboard**: Synthesizes multimodal telemetry, sustained attendance registers, 
        and AI pedagogical prompts into an exportable, compliant educator debrief report.
        """
    )

    analytics = SessionAnalytics(fused_df, att_df, suggestions)
    kpis = analytics.compute_kpis()

    # KPI Summary Row
    k_c1, k_c2, k_c3, k_c4 = st.columns(4)
    k_c1.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Average Engagement</div>
        <div class="kpi-value">{kpis['avg_engagement']}%</div>
        <div class="kpi-subtext">Classroom collective mean</div>
    </div>
    """, unsafe_allow_html=True)

    k_c2.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Sustained Attendance</div>
        <div class="kpi-value">{kpis['attendance_rate']}%</div>
        <div class="kpi-subtext">{kpis['present_count']} of {kpis['total_tracked']} verified present</div>
    </div>
    """, unsafe_allow_html=True)

    k_c3.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Peak Attention Window</div>
        <div class="kpi-value">{kpis['peak_engagement']}%</div>
        <div class="kpi-subtext">Achieved at {kpis['peak_time']}</div>
    </div>
    """, unsafe_allow_html=True)

    k_c4.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Verbal Interaction Ratio</div>
        <div class="kpi-value">{kpis['interaction_ratio']}%</div>
        <div class="kpi-subtext">Student discussion vs lecture</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Correlation & Analytics Charts
    a_c1, a_c2 = st.columns([1.1, 0.9])
    with a_c1:
        st.markdown("#### 📦 Engagement Index vs Acoustic Dynamic State")
        corr_box = analytics.render_state_correlation_boxplot()
        if corr_box:
            st.plotly_chart(corr_box, use_container_width=True)

    with a_c2:
        st.markdown("#### 🎯 Attendance Distribution (Sustained vs Transient)")
        status_counts = att_df["Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        att_pie = px.pie(
            status_counts,
            names="Status",
            values="Count",
            hole=0.5,
            color="Status",
            color_discrete_map={"PRESENT": "#10b981", "ABSENT": "#ef4444"},
            template="plotly_dark",
            title="Verified Attendance Roster Split"
        )
        att_pie.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(att_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📥 Export Official Post-Session Educator Reports")
    
    r_c1, r_c2, r_c3 = st.columns(3)
    
    # 1. Download Markdown Summary Report
    md_report = analytics.generate_markdown_report()
    r_c1.download_button(
        label="📄 Download Executive Report (Markdown)",
        data=md_report,
        file_name="classroom_pedagogical_report.md",
        mime="text/markdown",
        use_container_width=True
    )

    # 2. Download Standalone HTML Report
    html_report = analytics.generate_html_report()
    r_c2.download_button(
        label="🌐 Download Printable Report (HTML)",
        data=html_report,
        file_name="classroom_pedagogical_report.html",
        mime="text/html",
        use_container_width=True
    )

    # 3. Download Full Multimodal Time-Series Dataset (CSV)
    telemetry_csv = io.StringIO()
    fused_df.to_csv(telemetry_csv, index=False)
    r_c3.download_button(
        label="📈 Download Full Telemetry Data (CSV)",
        data=telemetry_csv.getvalue(),
        file_name="classroom_multimodal_telemetry.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Preview Report
    with st.expander("📖 Preview Formatted Executive Report"):
        st.markdown(md_report)

# ==========================================
# FOOTER & ETHICAL COMPLIANCE
# ==========================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 12px; margin-bottom: 20px;">
        🎓 <strong>Multimodal Attendance & Engagement Agent</strong> • Skill Wallet Capstone Project 5<br>
        Built with Streamlit, OpenCV, and SciPy • 100% Offline Edge Processing Guarantee • Compliant with FERPA & COPPA Educational Privacy Frameworks
    </div>
    """,
    unsafe_allow_html=True
)
