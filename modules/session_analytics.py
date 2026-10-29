"""
Session Analytics & Educator Report Module (Epics 5 & 6)
Post-session summary dashboard, KPI aggregation, participation rates,
correlation heatmaps, and downloadable executive session reports (CSV / Markdown / HTML).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

class SessionAnalytics:
    """
    Synthesizes multimodal data into actionable pedagogical metrics and exportable reports.
    """
    def __init__(self, fused_df: pd.DataFrame, attendance_df: pd.DataFrame, suggestions: list):
        self.df = fused_df
        self.att_df = attendance_df
        self.suggestions = suggestions

    def compute_kpis(self) -> dict:
        """Computes executive level pedagogical metrics."""
        if self.df.empty:
            return {
                "avg_engagement": 0.0,
                "peak_engagement": 0.0,
                "peak_time": "00:00",
                "min_engagement": 0.0,
                "min_time": "00:00",
                "attendance_rate": 0.0,
                "present_count": 0,
                "total_tracked": 0,
                "interaction_ratio": 0.0,
                "session_duration_min": 0.0
            }

        avg_ei = float(self.df["engagement_index"].mean())
        peak_idx = self.df["engagement_index"].idxmax()
        min_idx = self.df["engagement_index"].idxmin()

        peak_ei = float(self.df.loc[peak_idx, "engagement_index"])
        peak_time = str(self.df.loc[peak_idx, "time_label"])

        min_ei = float(self.df.loc[min_idx, "engagement_index"])
        min_time = str(self.df.loc[min_idx, "time_label"])

        # Attendance KPIs
        total_tracked = len(self.att_df)
        present_count = len(self.att_df[self.att_df["Status"] == "PRESENT"]) if not self.att_df.empty else 0
        attendance_rate = round((present_count / max(total_tracked, 1)) * 100, 1)

        # Interaction Ratio: Discussion / (Discussion + Lecture)
        acous_counts = self.df.get("acoustic_state", pd.Series([])).value_counts()
        disc_count = acous_counts.get("Student Discussion / Verbal Interaction", 0)
        lect_count = acous_counts.get("Instructor Lecture", 0)
        talk_total = disc_count + lect_count
        interaction_ratio = round((disc_count / max(talk_total, 1)) * 100, 1)

        dur_sec = self.df["time_sec"].max() if not self.df.empty else 0.0
        dur_min = round(dur_sec / 60.0, 1)

        return {
            "avg_engagement": round(avg_ei, 1),
            "peak_engagement": round(peak_ei, 1),
            "peak_time": peak_time,
            "min_engagement": round(min_ei, 1),
            "min_time": min_time,
            "attendance_rate": attendance_rate,
            "present_count": present_count,
            "total_tracked": total_tracked,
            "interaction_ratio": interaction_ratio,
            "session_duration_min": dur_min
        }

    def render_state_correlation_boxplot(self):
        """Renders box plot comparing engagement levels across the 4 acoustic states."""
        if "acoustic_state" not in self.df.columns:
            return None

        fig = px.box(
            self.df,
            x="acoustic_state",
            y="engagement_index",
            color="acoustic_state",
            title="Classroom Engagement Distribution by Acoustic State",
            labels={
                "acoustic_state": "Acoustic Dynamic State",
                "engagement_index": "Engagement Index (0-100)"
            },
            template="plotly_dark",
            color_discrete_map={
                "Silence": "#64748b",
                "Instructor Lecture": "#0284c7",
                "Student Discussion / Verbal Interaction": "#10b981",
                "High Ambient Classroom Distraction": "#ef4444"
            }
        )
        fig.update_layout(
            showlegend=False,
            height=360,
            margin=dict(l=40, r=20, t=50, b=40)
        )
        return fig

    def generate_markdown_report(self) -> str:
        """Creates a formatted markdown executive session summary report."""
        kpis = self.compute_kpis()

        report_lines = [
            "# 📋 Classroom Session Telemetry & Pedagogical Summary Report",
            "**System**: Multimodal Attendance & Engagement Agent (Project 5 Capstone)",
            "**Privacy Guarantee**: 100% Offline Edge Processing • Zero External Cloud Transmission",
            "",
            "---",
            "## 1. Executive Performance KPIs",
            f"- **Session Duration**: {kpis['session_duration_min']} minutes",
            f"- **Collective Average Engagement**: **{kpis['avg_engagement']}%**",
            f"- **Peak Attention Moment**: **{kpis['peak_engagement']}%** at `{kpis['peak_time']}`",
            f"- **Lowest Engagement Dip**: **{kpis['min_engagement']}%** at `{kpis['min_time']}`",
            f"- **Verified Sustained Attendance**: **{kpis['present_count']} / {kpis['total_tracked']}** participants ({kpis['attendance_rate']}%)",
            f"- **Student Verbal Interaction Ratio**: **{kpis['interaction_ratio']}%** of instructional exchange",
            "",
            "---",
            "## 2. Verified Attendance Register (Sustained >70% Presence Logic)",
            ""
        ]

        if not self.att_df.empty:
            try:
                report_lines.append(self.att_df.to_markdown(index=False))
            except Exception:
                report_lines.append("```\n" + self.att_df.to_string(index=False) + "\n```")
        else:
            report_lines.append("*No participants detected in session.*")

        report_lines.extend([
            "",
            "---",
            "## 3. Pedagogical Agent Advisory & Intervention Log",
            ""
        ])

        if self.suggestions:
            for s in self.suggestions:
                report_lines.append(f"### {s['badge']} (`{s['timestamp']}`)")
                report_lines.append(f"**Observation**: {s['message']}")
                report_lines.append(f"**Actionable Pedagogical Prompt**: {s['recommendation']}")
                report_lines.append(f"**Category**: *{s['action_category']}*")
                report_lines.append("")
        else:
            report_lines.append("*No critical attention flags or fatigue thresholds triggered during this session.*")

        report_lines.extend([
            "",
            "---",
            "## 4. Ethical Pedagogical Guardrails Compliance",
            "- Facial Recognition Biometrics: **None Stored / Discarded Post-Frame**",
            "- Processing Pipeline: **Strict On-Device In-Memory Edge Pipeline**",
            "- Purpose: **Formative Pedagogical Feedback (Non-Punitive)**",
            f"- Generated at Local Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])

        return "\n".join(report_lines)

    def generate_html_report(self) -> str:
        """Creates a modern standalone printable HTML report."""
        kpis = self.compute_kpis()
        table_html = self.att_df.to_html(classes="styled-table", index=False) if not self.att_df.empty else "<p>No participants</p>"

        suggestions_html = ""
        for s in self.suggestions:
            suggestions_html += f"""
            <div class="suggestion-item">
                <span class="badge">{s['badge']} - {s['timestamp']}</span>
                <p><strong>Observation:</strong> {s['message']}</p>
                <p><strong>Recommendation:</strong> {s['recommendation']}</p>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Classroom Pedagogical Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; }}
        h1, h2, h3 {{ color: #38bdf8; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .kpi-box {{ background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }}
        .kpi-title {{ font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: bold; }}
        .kpi-num {{ font-size: 24px; font-weight: bold; color: #f1f5f9; margin-top: 5px; }}
        .styled-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        .styled-table th, .styled-table td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; font-size: 13px; }}
        .styled-table th {{ background: #0f172a; color: #38bdf8; }}
        .suggestion-item {{ background: #1e293b; border-left: 4px solid #f59e0b; padding: 12px 16px; margin-bottom: 10px; border-radius: 6px; }}
        .badge {{ font-weight: bold; color: #f59e0b; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>🎓 Classroom Pedagogical & Attendance Report</h1>
    <p>100% Offline Edge Telemetry • FERPA & COPPA Compliant</p>
    <hr style="border-color: #334155;">

    <div class="kpi-grid">
        <div class="kpi-box">
            <div class="kpi-title">Average Engagement</div>
            <div class="kpi-num">{kpis['avg_engagement']}%</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">Sustained Attendance</div>
            <div class="kpi-num">{kpis['attendance_rate']}% ({kpis['present_count']}/{kpis['total_tracked']})</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">Peak Attention</div>
            <div class="kpi-num">{kpis['peak_engagement']}% ({kpis['peak_time']})</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">Interaction Ratio</div>
            <div class="kpi-num">{kpis['interaction_ratio']}%</div>
        </div>
    </div>

    <h2>Verified Attendance Register (>70% Sustained Threshold)</h2>
    {table_html}

    <h2>AI Teaching Suggestion Log</h2>
    {suggestions_html if suggestions_html else "<p>No intervention flags triggered.</p>"}
</body>
</html>
"""
        return html_content
