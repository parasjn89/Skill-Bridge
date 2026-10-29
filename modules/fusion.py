"""
Multimodal Fusion & Engagement Scoring Module (Epic 4)
Fuses visual attentiveness indicators (head pose, upright posture vs distraction/slumping)
and acoustic dynamics into a unified time-step matrix.
Computes normalized Engagement Index (0-100) and renders interactive timeline curves.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

class MultimodalFusionEngine:
    """
    Synchronizes temporal video telemetry and acoustic features into a unified matrix,
    evaluating collective classroom cognitive and behavioral engagement.
    """
    def __init__(self, weight_visual: float = 0.50, weight_audio: float = 0.35, weight_presence: float = 0.15):
        self.w_v = weight_visual
        self.w_a = weight_audio
        self.w_p = weight_presence
        # Normalize weights
        total = self.w_v + self.w_a + self.w_p
        if total > 0:
            self.w_v /= total
            self.w_a /= total
            self.w_p /= total

    def update_weights(self, weight_visual: float, weight_audio: float, weight_presence: float):
        total = weight_visual + weight_audio + weight_presence
        if total > 0:
            self.w_v = weight_visual / total
            self.w_a = weight_audio / total
            self.w_p = weight_presence / total

    def fuse(self, visual_df: pd.DataFrame, acoustic_df: pd.DataFrame, total_roster_count: int = 10) -> pd.DataFrame:
        """
        Merges visual time steps and acoustic windows on time_sec.
        Returns a rich multimodal time-series dataframe with normalized Engagement Index (0-100).
        """
        # Ensure time_sec alignment
        v_df = visual_df.copy()
        a_df = acoustic_df.copy()

        # Merge on nearest time_sec or round to nearest second
        v_df["time_sec_round"] = v_df["time_sec"].round().astype(int)
        a_df["time_sec_round"] = a_df["time_sec"].round().astype(int)

        merged = pd.merge(v_df, a_df, on="time_sec_round", how="outer", suffixes=("_vis", "_acous"))
        merged["time_sec"] = merged["time_sec_round"].astype(float)
        merged = merged.sort_values("time_sec").reset_index(drop=True)

        # Forward fill and backward fill missing values
        merged = merged.ffill().bfill()

        # Visual Attention Component V(t)
        if "attentive_count" in merged.columns and "total_visible" in merged.columns:
            vis_ratio = merged["attentive_count"] / merged["total_visible"].replace(0, 1)
            merged["visual_attention_score"] = (vis_ratio * 100.0).clip(0.0, 100.0)
        else:
            merged["visual_attention_score"] = 75.0

        # Audio Dynamics Component A(t)
        if "acoustic_engagement_score" not in merged.columns:
            merged["acoustic_engagement_score"] = 70.0

        # Presence Ratio Component P(t)
        if "total_visible" in merged.columns:
            pres_ratio = merged["total_visible"] / max(total_roster_count, 1)
            merged["presence_factor_score"] = (pres_ratio * 100.0).clip(0.0, 100.0)
        else:
            merged["presence_factor_score"] = 80.0

        # Raw Multimodal Engagement Index
        merged["raw_engagement_index"] = (
            self.w_v * merged["visual_attention_score"] +
            self.w_a * merged["acoustic_engagement_score"] +
            self.w_p * merged["presence_factor_score"]
        ).clip(0.0, 100.0)

        # Exponential Moving Average (EMA) to reflect collective cognitive inertia
        merged["engagement_index"] = merged["raw_engagement_index"].ewm(span=5, adjust=False).mean().round(1)

        # Classify Engagement State
        def classify_ei(val):
            if val >= 75.0:
                return "Optimal Flow (High)"
            elif val >= 50.0:
                return "Standard Focus (Moderate)"
            else:
                return "Attention Dip (Low)"

        merged["engagement_zone"] = merged["engagement_index"].apply(classify_ei)

        # Format MM:SS label
        merged["time_label"] = merged["time_sec"].apply(
            lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}"
        )

        # Detect Attention Peaks and Dips
        merged["is_peak"] = False
        merged["is_dip"] = False

        rolling_mean = merged["engagement_index"].rolling(window=7, min_periods=1, center=True).mean()
        for idx in range(1, len(merged) - 1):
            curr = merged.loc[idx, "engagement_index"]
            prev = merged.loc[idx - 1, "engagement_index"]
            nxt = merged.loc[idx + 1, "engagement_index"]

            # Peak condition
            if curr >= 78.0 and curr >= prev and curr >= nxt and curr > rolling_mean.iloc[idx] + 4:
                merged.loc[idx, "is_peak"] = True
            # Dip condition
            elif curr <= 48.0 and curr <= prev and curr <= nxt:
                merged.loc[idx, "is_dip"] = True

        return merged

    @staticmethod
    def render_engagement_timeline(df: pd.DataFrame):
        """
        Renders the comprehensive Multimodal Engagement Curve with green/yellow/red zones,
        peaks, dips, and synchronized component breakdowns.
        """
        fig = go.Figure()

        # 1. Background Colored Educational Zones
        max_t = df["time_sec"].max() if not df.empty else 100

        # Green Zone (Optimal: 75-100)
        fig.add_shape(type="rect", x0=0, y0=75, x1=max_t, y1=100,
                      fillcolor="rgba(16, 185, 129, 0.08)", line_width=0, layer="below")
        # Yellow Zone (Moderate: 50-75)
        fig.add_shape(type="rect", x0=0, y0=50, x1=max_t, y1=75,
                      fillcolor="rgba(245, 158, 11, 0.06)", line_width=0, layer="below")
        # Red Zone (Dip: 0-50)
        fig.add_shape(type="rect", x0=0, y0=0, x1=max_t, y1=50,
                      fillcolor="rgba(239, 68, 68, 0.08)", line_width=0, layer="below")

        # Threshold Line at 45% (Pedagogical Intervention Trigger)
        fig.add_shape(type="line", x0=0, y0=45, x1=max_t, y1=45,
                      line=dict(color="#ef4444", width=1.5, dash="dash"), layer="below")

        # 2. Main Normalized Engagement Index Curve
        fig.add_trace(go.Scatter(
            x=df["time_sec"],
            y=df["engagement_index"],
            mode="lines",
            name="Normalized Engagement Index (0-100)",
            line=dict(color="#38bdf8", width=3.5),
            hovertemplate="<b>Time: %{customdata[0]}</b><br>Engagement Index: <b>%{y:.1f}%</b><br>Zone: %{customdata[1]}<br>Acoustic State: %{customdata[2]}<extra></extra>",
            customdata=df[["time_label", "engagement_zone", "acoustic_state"]].values
        ))

        # 3. Sub-component Traces (Visual Attention & Audio Dynamics)
        fig.add_trace(go.Scatter(
            x=df["time_sec"],
            y=df["visual_attention_score"],
            mode="lines",
            name="Visual Attentive Pose (%)",
            line=dict(color="#a855f7", width=1.5, dash="dot"),
            opacity=0.7,
            hovertemplate="Time: %{customdata[0]}<br>Visual Score: %{y:.1f}%<extra></extra>",
            customdata=df[["time_label"]].values
        ))

        fig.add_trace(go.Scatter(
            x=df["time_sec"],
            y=df["acoustic_engagement_score"],
            mode="lines",
            name="Acoustic Energy Score (%)",
            line=dict(color="#10b981", width=1.5, dash="dash"),
            opacity=0.7,
            hovertemplate="Time: %{customdata[0]}<br>Audio Score: %{y:.1f}%<extra></extra>",
            customdata=df[["time_label"]].values
        ))

        # 4. Markers for Attention Peaks
        peaks = df[df["is_peak"] == True]
        if not peaks.empty:
            fig.add_trace(go.Scatter(
                x=peaks["time_sec"],
                y=peaks["engagement_index"],
                mode="markers+text",
                name="Attention Peak 🚀",
                marker=dict(color="#10b981", size=11, symbol="star"),
                text=["Peak" for _ in range(len(peaks))],
                textposition="top center",
                textfont=dict(color="#6ee7b7", size=10),
                hovertemplate="<b>🚀 Attention Peak</b><br>Time: %{customdata[0]}<br>Score: %{y:.1f}%<extra></extra>",
                customdata=peaks[["time_label"]].values
            ))

        # 5. Markers for Attention Dips
        dips = df[df["is_dip"] == True]
        if not dips.empty:
            fig.add_trace(go.Scatter(
                x=dips["time_sec"],
                y=dips["engagement_index"],
                mode="markers+text",
                name="Attention Dip ⚠️",
                marker=dict(color="#ef4444", size=11, symbol="x"),
                text=["Dip" for _ in range(len(dips))],
                textposition="bottom center",
                textfont=dict(color="#fca5a5", size=10),
                hovertemplate="<b>⚠️ Attention Dip</b><br>Time: %{customdata[0]}<br>Score: %{y:.1f}%<extra></extra>",
                customdata=dips[["time_label"]].values
            ))

        fig.update_layout(
            title="Multimodal Collective Engagement Timeline (Normalized Index 0 - 100)",
            xaxis_title="Session Elapsed Time (Seconds)",
            yaxis_title="Engagement Score (0 - 100)",
            yaxis=dict(range=[0, 105]),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=30, t=65, b=40),
            height=430,
            hovermode="x unified"
        )
        return fig
