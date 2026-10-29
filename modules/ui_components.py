"""
Educational Dashboard UI Components and Custom CSS
Provides cohesive, modern styling for the Multimodal Attendance & Engagement Agent.
"""

import streamlit as st

def apply_custom_css():
    """Injects high-contrast, clean educational dashboard styling."""
    st.markdown(
        """
        <style>
        /* Modern Educational SaaS Theme */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Hero Header Banner */
        .agent-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
            color: #f8fafc;
            padding: 24px 30px;
            border-radius: 14px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .agent-title {
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
            color: #ffffff;
        }
        
        .agent-subtitle {
            font-size: 14px;
            color: #94a3b8;
            margin-top: 6px;
            margin-bottom: 14px;
            font-weight: 400;
        }

        /* Pill Badges */
        .pill-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .badge-privacy {
            background-color: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }
        
        .badge-edge {
            background-color: rgba(56, 189, 248, 0.2);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.35);
        }
        
        .badge-pedagogy {
            background-color: rgba(168, 85, 247, 0.2);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.35);
        }

        /* Modern KPI Cards */
        .kpi-card {
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 18px 20px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.3);
            border-color: rgba(56, 189, 248, 0.3);
        }
        
        .kpi-label {
            font-size: 12px;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        
        .kpi-value {
            font-size: 26px;
            font-weight: 700;
            color: #f1f5f9;
        }
        
        .kpi-subtext {
            font-size: 12px;
            color: #64748b;
            margin-top: 4px;
        }

        /* Pedagogical Advice Cards */
        .pedagogy-card-urgent {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(239, 68, 68, 0.04));
            border-left: 4px solid #ef4444;
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 12px;
        }

        .pedagogy-card-advisory {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(245, 158, 11, 0.04));
            border-left: 4px solid #f59e0b;
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 12px;
        }

        .pedagogy-card-praise {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(16, 185, 129, 0.04));
            border-left: 4px solid #10b981;
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 12px;
        }

        /* Status Pills for Attendance Table */
        .status-present {
            background: #065f46;
            color: #6ee7b7;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
        }

        .status-absent {
            background: #831843;
            color: #fbcfe8;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
        }

        /* Streamlit Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 6px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 10px 18px;
            font-weight: 600;
            font-size: 14px;
            border-radius: 8px;
            color: #94a3b8;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(56, 189, 248, 0.15);
            color: #38bdf8 !important;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_hero_banner():
    """Renders the top application header with privacy architecture badges."""
    st.markdown(
        """
        <div class="agent-hero">
            <div class="agent-title">
                <span>🎓 Multimodal Attendance & Engagement Agent</span>
            </div>
            <div class="agent-subtitle">
                Skill Wallet Capstone Project 5 • Privacy-First Classroom Presence, Acoustic Dynamics & Multimodal Pedagogical Intelligence
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
                <span class="pill-badge badge-privacy">
                    🔒 100% Offline Edge Processing
                </span>
                <span class="pill-badge badge-edge">
                    ⚡ Zero External Cloud Media
                </span>
                <span class="pill-badge badge-pedagogy">
                    🤝 Collective Pedagogical Telemetry
                </span>
                <span class="pill-badge" style="background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3);">
                    📋 FERPA & COPPA Ethical Guardrails
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_privacy_callout():
    """Renders the privacy guarantee accordian / drawer."""
    with st.expander("🛡️ Privacy-First System Principles & Ethical Telemetry Guarantee (Epic 1)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                #### 🔒 100% Offline Edge Processing
                - **Zero Cloud Video Ingestion**: Every video frame and audio buffer is processed strictly in-memory on the local machine using OpenCV and NumPy.
                - **No Cloud Transmission**: Raw webcam, video frames, and audio streams never leave the runtime environment.
                - **Ephemeral In-Memory Buffers**: Frames are discarded immediately after feature extraction; no raw video is stored on disk.
                """
            )
        with c2:
            st.markdown(
                """
                #### 🤝 Ethical Pedagogical Telemetry
                - **Collective Over Punitive**: The system measures group-level pedagogical flow and attention peaks/dips to assist educators, not to conduct punitive individual surveillance.
                - **No Facial Recognition Biometrics**: Participants are assigned anonymous, ephemeral tracking tokens (`STU-01`, `STU-02`) based purely on spatial coordinates.
                - **Opt-In Face Blur**: Optional visual anonymization mode masks participant identities on display for pedagogical reports.
                """
            )
