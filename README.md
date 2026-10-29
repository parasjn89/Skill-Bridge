# 🎓 Multimodal Attendance & Engagement Agent
### Skill Wallet Capstone Project 5 • Production-Ready Streamlit Application

> **Privacy-First Classroom Intelligence**: 100% Offline Edge Processing • Zero External Cloud Video Transmission • Ethical Pedagogical Telemetry

---

## 🌟 Executive Overview

The **Multimodal Attendance & Engagement Agent** is a production-ready, edge-native classroom analytics platform designed to empower educators with real-time pedagogical insights while strictly protecting student privacy.

Traditional classroom surveillance systems rely on cloud video streaming and individual facial recognition biometrics—raising severe privacy, ethical, and legal violations (e.g., FERPA, COPPA, GDPR). In contrast, this agent implements **100% on-device edge computation**: raw video frames and audio buffers are processed entirely in local memory using OpenCV, SciPy, and NumPy. No biometric face templates are ever stored, and no raw media ever leaves the educator's device.

---

## 🏗️ System Architecture & Capstone Modules

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                LOCAL EDGE RUNTIME                                      │
├────────────────────────────────┬───────────────────────────────────────────────────────┤
│    VISUAL TELEMETRY STREAM     │               ACOUSTIC TELEMETRY STREAM               │
│  (Webcam / Video / Simulation) │             (Microphone / Waveform Buffer)            │
│               │                │                           │                           │
│               ▼                │                           ▼                           │
│   Haar Cascade Edge Detector   │            Sliding Window Time Segmentation           │
│   • Frontal & Profile Faces    │               (1.0s - 2.0s Time Windows)              │
│   • Eye / Head Pose Indicators │                           │                           │
│               │                │                           ▼                           │
│               ▼                │              Spectral Feature Extraction              │
│    Centroid Multi-Object       │           • RMS Energy & Decibels (dBFS)              │
│      Persistent Tracker        │           • Zero Crossing Rate (ZCR)                  │
│     (STU-01, STU-02...)        │           • Spectral Centroid (Pitch Center)          │
│               │                │           • Vocal Formant Band Ratio                  │
│               ▼                │                           │                           │
│   Sustained Presence Logic     │                           ▼                           │
│    (Threshold > 70% Time)      │                4-Class Acoustic VAD                   │
│   [Filters Pass-Throughs]      │            • Silence                                  │
│               │                │            • Instructor Lecture                       │
│               │                │            • Student Discussion                       │
│               │                │            • High Ambient Distraction                 │
└───────────────┼────────────────┴───────────────────────────┼───────────────────────────┘
                │                                            │
                └─────────────────────┬──────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │  MULTIMODAL FUSION ENGINE │
                        │  Normalized EI (0 - 100)  │
                        └─────────────┬─────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                ▼                                           ▼
┌───────────────────────────────┐           ┌───────────────────────────────┐
│  AI TEACHING SUGGESTION AGENT │           │  POST-SESSION ANALYTICS &     │
│   (Heuristic Pedagogical Nudge)│           │       EDUCATOR DEBRIEF        │
│ • Attention Dip Trigger (<45%) │           │ • Executive Session KPIs      │
│ • Monologue Fatigue (>15 min) │           │ • Verified Attendance (CSV)   │
│ • Distraction Re-anchoring    │           │ • Multimodal Telemetry (CSV)  │
│ • 2-Min Stretch Break Tool    │           │ • Formal Reports (MD / HTML)  │
└───────────────────────────────┘           └───────────────────────────────┘
```

---

## 🧩 Core Modules & Capstone Epics

### 1. Privacy-First Principles & Ethical Telemetry (Epic 1)
- **100% Offline Edge Processing**: Zero external cloud video transmission. All computer vision and audio digital signal processing (DSP) execute locally.
- **Ethical Telemetry Design**: Shifts the paradigm from punitive individual surveillance to collective group pedagogical feedback.
- **Anonymous Centroid Tracking**: Participants are assigned ephemeral session tokens (`STU-01`, `STU-02`) based purely on bounding box coordinates—no facial recognition biometrics or emotional profiling.
- **Edge Face Blurring**: One-click Gaussian blur anonymization for public classroom debriefs.

### 2. Tab 1 - Automated Presence-Based Attendance (Epic 2)
- **Video Feed Processor**: Supports live webcam snapshots, uploaded classroom videos/photos, or real-time multi-student interactive simulation.
- **Centroid Multi-Object Tracker**: Tracks participant coordinates across frames with occlusion resilience and profile-face handling.
- **Sustained Presence Logic**: Enforces a strict duration threshold (**>70% of active session time**) to prevent false attendance marks caused by students briefly walking past an open door (transient hallway pass-throughs).
- **Exportable Attendance Register**: Downloadable CSV with Student ID, First Seen (MM:SS), Last Seen (MM:SS), Total Duration, Presence %, and Status (`PRESENT` vs `ABSENT`).

### 3. Tab 2 - Acoustic Interaction & Noise Dynamics (Epic 3)
- **Audio Stream Processor**: Segments signals into sliding time windows (1.0s resolution).
- **Spectral Feature Extraction**: Computes RMS energy, dBFS levels, Zero Crossing Rate (ZCR), Spectral Centroid, and Vocal Band Formant Energy.
- **4-Class Acoustic State Classification**:
  1. `Silence`: Background ambient room tone.
  2. `Instructor Lecture`: Sustained vocal formant resonance (500–2500 Hz), low-variance envelope.
  3. `Student Discussion / Verbal Interaction`: High turn-taking dynamics, multi-speaker conversational bursts.
  4. `High Ambient Classroom Distraction`: High-frequency chaotic screeching, chair scraping, or loud disruptions.
- **Interactive Visualizations**: Decibel energy ribbon over time, acoustic distribution donut chart, and raw spectral window inspector.

### 4. Tab 3 - Multimodal Fusion & Engagement Scoring (Epic 4)
- **Multimodal Fusion Formula**:
  $$\text{Engagement Index}(t) = w_v \cdot V(t) + w_a \cdot A(t) + w_p \cdot P(t)$$
  - $V(t)$: Visual attentiveness ratio (forward head pose vs turned/slumped).
  - $A(t)$: Acoustic interaction score (Discussion: 90%, Lecture: 72%, Silence: 58%, Distraction: 25%).
  - $P(t)$: Collective participation roster ratio.
- **Exponential Moving Average (EMA)**: Models collective psychological inertia so indices reflect natural classroom attention flow.
- **Colored Educational Safety Zones**:
  - 🟢 **Optimal Flow (75 - 100%)**: High collaborative learning state.
  - 🟡 **Standard Focus (50 - 75%)**: Attentive instructional reception.
  - 🔴 **Critical Dip (< 50%)**: Flags intervention requirement.
- **Automatic Peak & Dip Detection**: Detects breakthroughs (🚀) and attention lulls (⚠️).

### 5. Tab 4 - AI Teaching Suggestion Engine (Epic 5)
- **Heuristic Advisory Agent**: Monitors temporal trends and generates non-intrusive pedagogical prompts:
  - *Collective Engagement Dip*: "Collective engagement dropped below 45% (Current: 39%) for 4 minutes — recommend launching a quick poll, interactive discussion, or a 2-minute stretch break."
  - *Instructor Monologue Fatigue*: "Continuous monologue detected for 18 minutes with 0% student interaction. Pause for a check-for-understanding."
  - *Distraction Spike*: "Acoustic distraction exceeded threshold (68 dBFS). Re-anchor classroom focus."
  - *Posture Slump*: "Visual indicators show physical lethargy. Guide a quick stand-up stretch."
  - *Flow State Peak*: "Engagement peaked at 92%! Ideal moment to solidify core takeaways."
- **Interactive Educator Toolkit**:
  - Built-in 2-minute Guided Stretch Break timer.
  - Quick-Launch Formative Assessment templates (60s Poll, Think-Pair-Share, Muddiest Point).

### 6. Tab 5 - Session Analytics & Educator Report (Epics 5 & 6)
- **Post-Session Executive Dashboard**: Average Engagement Index, Sustained Attendance Rate, Peak Attention Moment, Student Verbal Interaction Ratio.
- **Acoustic-Engagement Correlation Matrix**: Box plot distribution across acoustic states.
- **1-Click Report Downloads**:
  - `classroom_pedagogical_report.md`: Complete Executive Report in Markdown.
  - `classroom_pedagogical_report.html`: Standalone Printable HTML Report.
  - `verified_attendance_register.csv`: Compliant attendance table.
  - `classroom_multimodal_telemetry.csv`: Full time-series dataset.

---

## 🚀 Quickstart & Local Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Automated Test Suite
```bash
python test_agent_pipeline.py
```

---

## ☁️ 1-Click Streamlit Community Cloud Deployment

This application is strictly designed with **zero external database dependencies** (`sqlite`, `postgres`, etc.) and bundled edge models:
1. Push this repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io).
3. Select your repository, set the main file path to `app.py`, and click **Deploy**.
4. The application boots instantly using bundled Haar cascade models and on-device signal processing.

---

## 🔒 Compliance & Ethical Guardrails

- **FERPA Compliance (Family Educational Rights and Privacy Act)**: No identifiable student records or face embeddings are transmitted over networks or saved to persistent external databases.
- **COPPA Compliance (Children's Online Privacy Protection Act)**: Zero cloud ingestion of minor video or voice data.
- **Pedagogical Telemetry Mandate**: Telemetry metrics are designed strictly for teacher self-reflection and pedagogical optimization, never for punitive student evaluation.
