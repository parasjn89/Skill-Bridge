"""
Acoustic Interaction & Noise Dynamics Module (Epic 3)
Audio Stream Processor, Spectral Energy Extraction, and Voice Activity Detection (VAD)
Classifying audio into Silence, Instructor Lecture, Student Discussion, and High Distraction.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfft, rfftfreq
import plotly.graph_objects as go
import plotly.express as px

class AcousticAnalyzer:
    """
    Offline Acoustic Signal Processor & 4-Class Classroom Dynamics Classifier.
    Guarantees 100% on-device feature extraction without sending audio buffers to any cloud API.
    """
    CLASSES = [
        "Silence",
        "Instructor Lecture",
        "Student Discussion / Verbal Interaction",
        "High Ambient Classroom Distraction"
    ]

    CLASS_COLORS = {
        "Silence": "#64748b",                                # Slate Gray
        "Instructor Lecture": "#0284c7",                     # Ocean Blue
        "Student Discussion / Verbal Interaction": "#10b981",# Emerald Green
        "High Ambient Classroom Distraction": "#ef4444"      # Crimson Red
    }

    def __init__(self, sample_rate: int = 16000, window_sec: float = 1.0):
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.samples_per_window = int(sample_rate * window_sec)

    def extract_window_features(self, window: np.ndarray):
        """
        Extracts spectral energy, zero crossing rate, spectral centroid, and voice band ratio.
        """
        if len(window) == 0:
            return 0.0, -100.0, 0.0, 0.0, 0.0, 0.0

        # 1. RMS Energy & dBFS
        rms = np.sqrt(np.mean(window**2) + 1e-12)
        db = 20 * np.log10(rms + 1e-9)

        # 2. Zero Crossing Rate (ZCR)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(window)))) / (2 * len(window))

        # 3. Frequency Spectrum & Spectral Centroid
        spectrum = np.abs(rfft(window * np.hanning(len(window))))
        freqs = rfftfreq(len(window), d=1.0 / self.sample_rate)

        sum_spectrum = np.sum(spectrum) + 1e-10
        spectral_centroid = np.sum(freqs * spectrum) / sum_spectrum

        # 4. Voice Frequency Band (300 Hz - 3200 Hz) vs High Frequency Band (> 3500 Hz)
        voice_mask = (freqs >= 300) & (freqs <= 3200)
        high_mask = (freqs > 3500)

        voice_energy = np.sum(spectrum[voice_mask]) / sum_spectrum
        high_energy = np.sum(spectrum[high_mask]) / sum_spectrum

        # 5. Envelope Variance (Turn-taking volatility)
        sub_windows = np.array_split(window, 8)
        sub_rms = [np.sqrt(np.mean(sw**2) + 1e-12) for sw in sub_windows]
        envelope_var = float(np.std(sub_rms) / (np.mean(sub_rms) + 1e-6))

        return rms, db, zero_crossings, spectral_centroid, voice_energy, envelope_var

    def classify_window(self, db: float, zcr: float, centroid: float, voice_energy: float, env_var: float):
        """
        Classifies audio window into 4 educational acoustic states using deterministic physical acoustic rules.
        """
        # Threshold 1: Silence
        if db < -42.0:
            return "Silence"

        # Threshold 2: High Ambient Distraction (scraping chairs, chaotic shuffling, yelling, outside sirens)
        if (db > -18.0 and centroid > 3200) or (zcr > 0.18 and voice_energy < 0.35) or (db > -14.0 and env_var > 1.2):
            return "High Ambient Classroom Distraction"

        # Threshold 3: Student Discussion / Verbal Interaction
        # Characterized by multi-speaker bursts, higher envelope variance, active voice band
        if voice_energy > 0.45 and env_var > 0.45:
            return "Student Discussion / Verbal Interaction"

        # Threshold 4: Instructor Lecture
        # Steadily sustained human voice band, consistent fundamental formants, moderate envelope variance
        if voice_energy > 0.40 and 400 <= centroid <= 2600:
            return "Instructor Lecture"

        # Fallback based on energy
        if db < -32.0:
            return "Silence"
        elif voice_energy > 0.35:
            return "Instructor Lecture"
        else:
            return "High Ambient Classroom Distraction"

    def process_audio_signal(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """
        Processes entire audio array into a time-series acoustic telemetry dataframe.
        """
        if sample_rate != self.sample_rate:
            # Resample or adapt
            self.sample_rate = sample_rate
            self.samples_per_window = int(sample_rate * self.window_sec)

        # Convert to mono float32 if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)

        # Normalize if max amplitude > 1.0
        max_amp = np.max(np.abs(audio_data))
        if max_amp > 1.0:
            audio_data = audio_data / max_amp

        total_samples = len(audio_data)
        num_windows = max(1, total_samples // self.samples_per_window)

        records = []
        for w_idx in range(num_windows):
            start_idx = w_idx * self.samples_per_window
            end_idx = min(start_idx + self.samples_per_window, total_samples)
            window = audio_data[start_idx:end_idx]

            time_sec = round(w_idx * self.window_sec, 2)
            rms, db, zcr, centroid, v_energy, env_var = self.extract_window_features(window)
            classification = self.classify_window(db, zcr, centroid, v_energy, env_var)

            # Map to an Acoustic Engagement Score (0 - 100)
            if classification == "Student Discussion / Verbal Interaction":
                acoustic_eng_score = 90.0
            elif classification == "Instructor Lecture":
                acoustic_eng_score = 72.0
            elif classification == "Silence":
                acoustic_eng_score = 58.0 # Reflective silence or individual work
            else: # Distraction
                acoustic_eng_score = 25.0

            records.append({
                "time_sec": time_sec,
                "time_label": f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}",
                "rms": round(rms, 4),
                "dbfs": round(db, 1),
                "zcr": round(zcr, 3),
                "spectral_centroid": round(centroid, 1),
                "voice_energy_ratio": round(v_energy, 3),
                "envelope_var": round(env_var, 3),
                "acoustic_state": classification,
                "acoustic_engagement_score": acoustic_eng_score
            })

        df = pd.DataFrame(records)
        return df

    @staticmethod
    def render_timeline_plot(df: pd.DataFrame):
        """Generates an interactive Plotly timeline chart for acoustic dynamics."""
        fig = go.Figure()

        # Decibel Level Area / Line
        fig.add_trace(go.Scatter(
            x=df["time_sec"],
            y=df["dbfs"],
            mode="lines",
            name="Audio Energy (dBFS)",
            line=dict(color="#38bdf8", width=2),
            hovertemplate="Time: %{customdata[0]}<br>Energy: %{y:.1f} dBFS<br>State: %{customdata[1]}<extra></extra>",
            customdata=df[["time_label", "acoustic_state"]].values
        ))

        # Colored background spans or markers for states
        for state, color in AcousticAnalyzer.CLASS_COLORS.items():
            state_sub = df[df["acoustic_state"] == state]
            if not state_sub.empty:
                fig.add_trace(go.Scatter(
                    x=state_sub["time_sec"],
                    y=state_sub["dbfs"],
                    mode="markers",
                    name=state,
                    marker=dict(color=color, size=6, symbol="circle"),
                    hovertemplate=f"<b>{state}</b><br>Time: %{{customdata[0]}}<br>Energy: %{{y:.1f}} dBFS<extra></extra>",
                    customdata=state_sub[["time_label"]].values
                ))

        fig.update_layout(
            title="Classroom Acoustic Dynamics & Energy Ribbon Over Time",
            xaxis_title="Session Elapsed Time (Seconds)",
            yaxis_title="Spectral Energy (dBFS)",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=30, t=60, b=40),
            height=380,
            hovermode="closest"
        )
        return fig

    @staticmethod
    def render_distribution_pie(df: pd.DataFrame):
        """Renders acoustic state percentage distribution pie / donut chart."""
        counts = df["acoustic_state"].value_counts().reset_index()
        counts.columns = ["State", "Windows"]
        counts["Percentage"] = (counts["Windows"] / len(df) * 100).round(1)

        colors = [AcousticAnalyzer.CLASS_COLORS.get(s, "#94a3b8") for s in counts["State"]]

        fig = go.Figure(data=[go.Pie(
            labels=counts["State"],
            values=counts["Windows"],
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hoverinfo="label+value+percent",
            textposition="inside"
        )])

        fig.update_layout(
            title="Acoustic Distribution of Session Time",
            template="plotly_dark",
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
            height=340
        )
        return fig
