import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import tempfile
from fpdf import FPDF
import base64

# Forecasting
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except:
    ExponentialSmoothing = None

# Decomposition
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except:
    seasonal_decompose = None

# Local anomaly detection
try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None

st.set_page_config(page_title="Anomaly Dashboard + Gemini Copilot", layout="wide")

BACKEND_URL = "http://127.0.0.1:8000/analyze"

# ----------------------------------
# ✅ NAVIGATION SIDEBAR
# ----------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", [
    "Upload CSV",
    "Detected Anomalies",
    "Graphs",
    "Forecasting",
    "Trend Decomposition",
    "Custom Severity Builder",
    "Recommendations",
    "Export PDF",
    "Gemini Chat"
])

# ----------------------------------
# ✅ STORAGE (session state)
# ----------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "anomalies" not in st.session_state:
    st.session_state.anomalies = None

if "time_col" not in st.session_state:
    st.session_state.time_col = None

if "metric_col" not in st.session_state:
    st.session_state.metric_col = None

if "local_anomaly_preview" not in st.session_state:
    st.session_state.local_anomaly_preview = None

# =====================================================================================
# Utility: local anomaly detector
# =====================================================================================
def detect_anomalies_local(df, time_col, metric_col, rolling_window=5, contamination=0.1):
    if IsolationForest is None:
        raise RuntimeError("sklearn is not available. Install scikit-learn.")

    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    tmp = tmp.sort_values(time_col).reset_index(drop=True)

    tmp["rolling_mean"] = tmp[metric_col].rolling(window=rolling_window, min_periods=1).mean()
    tmp["residual"] = tmp[metric_col] - tmp["rolling_mean"]

    tmp["resid_mean"] = tmp["residual"].mean()
    tmp["resid_std"] = tmp["residual"].std() if tmp["residual"].std() > 0 else 1e-9
    tmp["zscore"] = (tmp["residual"] - tmp["resid_mean"]) / tmp["resid_std"]

    m = IsolationForest(contamination=contamination, random_state=42)
    m.fit(tmp[["residual", "zscore"]])
    tmp["score"] = -m.decision_function(tmp[["residual", "zscore"]])

    tmp = tmp.drop(columns=["resid_mean", "resid_std"], errors="ignore")
    return tmp

def apply_severity_from_thresholds(df_scores, thresholds):
    df = df_scores.copy()
    def sev(s):
        if s > thresholds["critical"]:
            return "Critical"
        elif s > thresholds["high"]:
            return "High"
        elif s > thresholds["medium"]:
            return "Medium"
        else:
            return "Low"
    df["severity"] = df["score"].apply(sev)
    df["is_anomaly"] = df["severity"] != "Low"
    return df

# =====================================================================================
# PAGE 1 — UPLOAD CSV
# =====================================================================================
if page == "Upload CSV":
    st.title("📊 Upload Your CSV File")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.session_state.df = None

    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        time_col = st.selectbox("Select Time Column", df.columns, index=0)
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in dataset.")
        else:
            metric_col = st.selectbox("Select Numeric Column", numeric_cols, index=0)

            st.session_state.time_col = time_col
            st.session_state.metric_col = metric_col

            # Automatic local anomaly detection
            try:
                local_df = detect_anomalies_local(df, time_col, metric_col)
                thresholds = {
                    "medium": np.percentile(local_df["score"], 75),
                    "high": np.percentile(local_df["score"], 90),
                    "critical": np.percentile(local_df["score"], 97)
                }
                preview_df = apply_severity_from_thresholds(local_df, thresholds)

                st.session_state.local_anomaly_preview = {
                    "df": preview_df,
                    "thresholds": thresholds,
                    "rolling_window": 5,
                    "contamination": 0.1,
                    "method": "Percentile (automatic)"
                }

                st.session_state.anomalies = [
                    {
                        "timestamp": str(r.name),
                        "value": float(r[metric_col]),
                        "rolling_mean": float(r["rolling_mean"]),
                        "residual": float(r["residual"]),
                        "score": float(r["score"]),
                        "severity": r["severity"],
                        "root_causes_ranked": []
                    }
                    for idx, r in preview_df.iterrows()
                ]
                st.success(f"✅ Local anomalies detected automatically: {preview_df['is_anomaly'].sum()} anomalies.")
            except Exception as e:
                st.error(f"Automatic anomaly detection failed: {e}")

# =====================================================================================
# PAGE 2 — DETECTED ANOMALIES
# =====================================================================================
elif page == "Detected Anomalies":
    st.title("🔎 Detected Anomalies")
    anomalies = st.session_state.anomalies
    if anomalies is None:
        st.info("Please upload a CSV and run detection first.")
    else:
        for a in anomalies:
            st.markdown(f"""
            ### 🔴 Timestamp: {a.get('timestamp', 'N/A')}
            **Severity:** {a.get('severity', 'N/A')}  
            **Actual Value:** {a.get('value', 'N/A')}  
            **Rolling Mean:** {a.get('rolling_mean', 'N/A')}  
            **Residual:** {a.get('residual', 'N/A')}  
            **Score:** {a.get('score', 'N/A')}
            """)
            st.markdown("**Root Causes:**")
            rc = a.get("root_causes_ranked", [])
            if isinstance(rc, list):
                for cause, conf in rc:
                    st.write(f"- {cause} ({int(conf*100)}%)")
            else:
                st.write(rc)
            st.markdown("---")

# =====================================================================================
# PAGE 3 — GRAPHS
# =====================================================================================
elif page == "Graphs":
    st.title("📈 Anomaly Graphs")
    df = st.session_state.df
    time_col = st.session_state.time_col
    metric_col = st.session_state.metric_col
    if df is None:
        st.info("Upload data first.")
    else:
        try:
            df_plot = df.copy()
            df_plot[time_col] = pd.to_datetime(df_plot[time_col])
            df_plot = df_plot.sort_values(time_col)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_plot[time_col], df_plot[metric_col], linewidth=2, label="Metric")
            if st.session_state.anomalies:
                for a in st.session_state.anomalies:
                    try:
                        ts = pd.to_datetime(a.get("timestamp"))
                        val = a.get("value")
                        if pd.notna(ts) and val is not None:
                            ax.scatter([ts], [val], color="red", zorder=5)
                    except Exception:
                        pass
            ax.set_title("Time Series")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Graphing error: {e}")

# =====================================================================================
# PAGE 4 — FORECASTING
# =====================================================================================
elif page == "Forecasting":
    st.title("🔮 Forecasting & Future Predictions")

    df = st.session_state.df
    time_col = st.session_state.time_col
    metric_col = st.session_state.metric_col

    if df is None:
        st.info("Upload CSV first.")
    else:
        df_fc = df.copy()
        try:
            df_fc[time_col] = pd.to_datetime(df_fc[time_col])
        except Exception:
            df_fc[time_col] = pd.to_datetime(df_fc[time_col], errors='coerce')
        df_fc = df_fc.sort_values(time_col)
        df_fc.set_index(time_col, inplace=True)

        ts = df_fc[metric_col].dropna()
        periods = st.slider("Forecast Horizon (days)", 7, 60, 30)

        if ExponentialSmoothing is None:
            st.info("statsmodels not installed. Forecasting fallback will be linear extrapolation.")
        if st.button("🔮 Generate Forecast"):
            with st.spinner("Forecasting..."):
                # Holt-Winters
                if ExponentialSmoothing is not None and len(ts) >= 3:
                    try:
                        model = ExponentialSmoothing(ts, trend="add", seasonal=None)
                        fit = model.fit()
                        forecast = fit.forecast(periods)
                        st.success("✅ Forecast generated (Holt-Winters).")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(ts.index, ts.values, label="Actual", linewidth=2)
                        ax.plot(forecast.index, forecast.values, label="Forecast", linestyle="--")
                        ax.set_title("Actual vs Forecast")
                        ax.legend()
                        st.pyplot(fig)
                        forecast_df = pd.DataFrame({"date": forecast.index, "forecast": forecast.values})
                        csv = forecast_df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
                        st.dataframe(forecast_df)
                    except Exception as e:
                        st.error(f"Holt-Winters forecasting failed: {e}")
                else:
                    # fallback linear trend
                    try:
                        n = len(ts)
                        if n < 3:
                            last = ts.iloc[-1]
                            fc_vals = np.array([last] * periods)
                            fc_dates = pd.date_range(start=pd.to_datetime(ts.index[-1]) + pd.Timedelta(days=1), periods=periods, freq='D')
                        else:
                            N = min(14, max(3, n // 4))
                            recent = ts.iloc[-N:]
                            x = np.arange(len(recent))
                            y = recent.values
                            a, b = np.polyfit(x, y, 1)
                            fc_x = np.arange(len(recent), len(recent) + periods)
                            fc_vals = a * fc_x + b
                            last_date = pd.to_datetime(recent.index[-1])
                            fc_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(ts.index, ts.values, label="Actual", linewidth=2)
                        ax.plot(fc_dates, fc_vals, label="Forecast (linear)", linestyle="--")
                        ax.set_title("Actual vs Forecast (fallback)")
                        ax.legend()
                        st.pyplot(fig)
                        forecast_df = pd.DataFrame({"date": fc_dates, "forecast": np.round(fc_vals, 4)})
                        csv = forecast_df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download Forecast CSV (fallback)", data=csv, file_name="forecast_fallback.csv", mime="text/csv")
                        st.dataframe(forecast_df)
                    except Exception as e:
                        st.error(f"Fallback forecasting error: {e}")

# =====================================================================================
# PAGE 5 — TREND DECOMPOSITION
# =====================================================================================
elif page == "Trend Decomposition":
    st.title("📊 Trend Decomposition (Trend + Seasonality + Noise)")

    df = st.session_state.df
    tcol = st.session_state.time_col
    metric = st.session_state.metric_col

    if df is None:
        st.info("Upload CSV first.")
    else:
        df2 = df.copy()
        try:
            df2[tcol] = pd.to_datetime(df2[tcol])
        except Exception:
            df2[tcol] = pd.RangeIndex(len(df2))
        df2 = df2.sort_values(tcol).reset_index(drop=True)
        df2.set_index(tcol, inplace=True)
        ts = df2[metric].copy()

        default_period = 7
        period = st.number_input("Seasonal period (data frequency units, default 7)", min_value=1, value=default_period, step=1)

        nobs = ts.dropna().shape[0]

        if seasonal_decompose is not None and nobs >= 2 * period:
            try:
                ts_freq = ts.asfreq('D')
                result = seasonal_decompose(ts_freq, model="additive", period=period, extrapolate_trend='freq')
                # Plot trend, seasonality, residual
                for title, series in [("Trend", result.trend), ("Seasonality", result.seasonal), ("Residual", result.resid)]:
                    fig, ax = plt.subplots(figsize=(12, 3))
                    ax.plot(series)
                    ax.set_title(title)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Decomposition failed: {e}")
        else:
            st.info("Using fallback decomposition (rolling-trend + estimated seasonality). Works with short series.")
            try:
                window = max(3, min(nobs-1, int(max(3, nobs//4))))
                trend = ts.rolling(window=window, center=True, min_periods=1).mean()
                seasonal = ts - trend
                residual = ts - trend - seasonal
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(trend); ax.set_title("Trend (fallback)"); st.pyplot(fig)
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(seasonal); ax.set_title("Seasonality (fallback)"); st.pyplot(fig)
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(residual); ax.set_title("Residual (fallback)"); st.pyplot(fig)
            except Exception as e:
                st.error(f"Fallback decomposition error: {e}")

# =====================================================================================
# PAGE 6 — CUSTOM SEVERITY BUILDER
# =====================================================================================
elif page == "Custom Severity Builder":
    st.title("⚙️ Custom Severity Logic Builder")

    df = st.session_state.df
    if df is None:
        st.info("Upload CSV first (Upload CSV page).")
    else:
        time_col = st.session_state.time_col
        metric_col = st.session_state.metric_col

        st.markdown("### 1) Local detection settings")
        col1, col2 = st.columns(2)
        with col1:
            rolling_window = st.number_input("Rolling window (residual)", min_value=1, max_value=365, value=5)
            contamination = st.slider("IsolationForest contamination", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        with col2:
            method = st.selectbox("Threshold type", ["Percentile", "Absolute values", "Std-based"])

        st.markdown("### 2) Threshold settings")
        if method=="Percentile":
            p_med = st.number_input("Medium percentile", 0, 100, 75)
            p_high = st.number_input("High percentile", 0, 100, 90)
            p_crit = st.number_input("Critical percentile", 0, 100, 97)
        elif method=="Absolute values":
            a_med = st.number_input("Medium cutoff", value=0.01, format="%.6f")
            a_high = st.number_input("High cutoff", value=0.03, format="%.6f")
            a_crit = st.number_input("Critical cutoff", value=0.1, format="%.6f")
        else:
            k_med = st.number_input("Medium = mean + k*std", 0.0, value=0.5)
            k_high = st.number_input("High = mean + k*std", 0.0, value=1.0)
            k_crit = st.number_input("Critical = mean + k*std", 0.0, value=1.5)

        if st.button("Run local detection & preview"):
            local_df = detect_anomalies_local(df, time_col, metric_col, rolling_window, contamination)
            if method=="Percentile":
                thresholds = {"medium": np.percentile(local_df["score"],p_med),
                              "high": np.percentile(local_df["score"],p_high),
                              "critical": np.percentile(local_df["score"],p_crit)}
            elif method=="Absolute values":
                thresholds = {"medium": a_med, "high": a_high, "critical": a_crit}
            else:
                mean_score = local_df["score"].mean()
                std_score = local_df["score"].std() if local_df["score"].std()>0 else 1e-9
                thresholds = {"medium": mean_score+k_med*std_score,
                              "high": mean_score+k_high*std_score,
                              "critical": mean_score+k_crit*std_score}
            preview_df = apply_severity_from_thresholds(local_df, thresholds)
            st.session_state.local_anomaly_preview = {"df": preview_df,"thresholds": thresholds,
                                                       "rolling_window": rolling_window,"contamination": contamination,
                                                       "method": method}
            st.success(f"Preview ready — {preview_df['is_anomaly'].sum()} anomalies")
            st.dataframe(preview_df[[time_col, metric_col, "rolling_mean","residual","score","severity","is_anomaly"]].head(200))

        if st.session_state.local_anomaly_preview:
            if st.button("Apply severity mapping to dashboard anomalies"):
                preview = st.session_state.local_anomaly_preview
                df_preview = preview["df"]
                applied = []
                for idx,r in df_preview.iterrows():
                    applied.append({
                        "timestamp": str(r[time_col]),
                        "value": float(r[metric_col]),
                        "rolling_mean": float(r["rolling_mean"]),
                        "residual": float(r["residual"]),
                        "score": float(r["score"]),
                        "severity": r["severity"],
                        "root_causes_ranked": []
                    })
                st.session_state.anomalies = applied
                st.success("✅ Applied to dashboard anomalies!")

# =====================================================================================
# PAGE 7 — RECOMMENDATIONS
# =====================================================================================
# =====================================================================================
# PAGE 7 — RECOMMENDATIONS
# =====================================================================================
elif page == "Recommendations":
    st.title("💡 AI Recommendations & Insights")

    anomalies = st.session_state.anomalies
    if anomalies is None:
        st.info("No anomalies detected yet. Please upload CSV and detect anomalies first.")
    else:
        st.markdown("### Summary of Detected Anomalies")
        severity_counts = {"Low":0, "Medium":0, "High":0, "Critical":0}
        for a in anomalies:
            sev = a.get("severity", "Low")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        st.write(f"📊 **Low:** {severity_counts['Low']}, **Medium:** {severity_counts['Medium']}, **High:** {severity_counts['High']}, **Critical:** {severity_counts['Critical']}")

        st.markdown("---")
        st.markdown("### Recommendations Based on Severity")

        for a in anomalies:
            ts = a.get("timestamp", "N/A")
            val = a.get("value", "N/A")
            sev = a.get("severity", "Low")

            st.markdown(f"#### 🔴 Timestamp: {ts} — Severity: {sev}")
            st.write(f"**Value:** {val}")
            
            if sev == "Critical":
                st.info("⚠️ **Critical:** Immediate action required! Investigate the root cause and apply corrective measures immediately.")
            elif sev == "High":
                st.warning("⚠️ **High:** Needs attention soon. Check potential causes and monitor closely.")
            elif sev == "Medium":
                st.success("✅ **Medium:** Moderate anomaly. Keep an eye on trends and consider minor adjustments.")
            else:
                st.write("ℹ️ **Low:** No action needed. Normal fluctuation.")

            # Root cause suggestions
            rc = a.get("root_causes_ranked", [])
            if rc:
                st.markdown("**Possible Root Causes:**")
                for cause, conf in rc:
                    st.write(f"- {cause} ({int(conf*100)}% confidence)")
            else:
                st.write("**Root causes not identified yet.**")

            st.markdown("---")

        st.markdown("### General Advice:")
        st.write("""
        - Regularly monitor metrics to detect anomalies early.
        - Tune thresholds in **Custom Severity Builder** for better sensitivity.
        - Use forecasting and trend decomposition to anticipate potential spikes.
        - Combine anomaly insights with domain knowledge for more accurate decisions.
        """)


# =====================================================================================
# PAGE 8 — EXPORT PDF
# =====================================================================================
elif page == "Export PDF":
    st.title("📝 Export Report PDF")
    anomalies = st.session_state.anomalies
    if anomalies is None:
        st.info("No anomalies yet. Run detection first.")
    else:
        if st.button("Generate PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Anomaly Report", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            for a in anomalies:
                pdf.ln(5)
                pdf.multi_cell(0, 6, f"Timestamp: {a.get('timestamp','N/A')}\nValue: {a.get('value','N/A')}\nSeverity: {a.get('severity','N/A')}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                pdf.output(tmpfile.name)
                tmpfile.seek(0)
                st.download_button("📥 Download PDF", data=open(tmpfile.name,"rb").read(), file_name="anomaly_report.pdf")

# =====================================================================================
# PAGE 9 — GEMINI CHAT
# =====================================================================================
# =====================================================================================
# ✅ PAGE 6 — GEMINI CHAT (Integrated Chatbot)
# =====================================================================================
elif page == "Gemini Chat":
    st.title("🤖 Gemini AI Chatbot")

    df = st.session_state.df
    anomalies = st.session_state.anomalies

    if df is None:
        st.info("Upload CSV and run anomaly detection first.")
    else:

        # ------------------------------
        # ✅ Dataset Summary for Context
        # ------------------------------
        if anomalies:
            dataset_summary = "Detected anomalies:\n"
            for a in anomalies[:20]:
                dataset_summary += (
                    f"- {a['timestamp']} | Severity: {a['severity']} | "
                    f"Value: {a['value']}\n"
                )
        else:
            dataset_summary = (
                f"Dataset rows: {len(df)}\n"
                f"Columns: {list(df.columns)}\n"
                f"Numeric summary: {df.describe().to_dict()}"
            )

        # ------------------------------
        # ✅ Initialize Chat Memory
        # ------------------------------
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # ------------------------------
        # ✅ Display Chat Messages
        # ------------------------------
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ------------------------------
        # ✅ Chat Input Box
        # ------------------------------
        user_msg = st.chat_input("Ask about anomalies, trends, or dataset insights...")

        if user_msg:
            # Save user message
            st.session_state.chat_history.append(
                {"role": "user", "content": user_msg}
            )

            # Display immediately
            with st.chat_message("user"):
                st.markdown(user_msg)

            # Send to backend
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        res = requests.post(
                            "http://127.0.0.1:8000/chat",
                            json={
                                "question": user_msg,
                                "dataset_summary": dataset_summary
                            },
                            timeout=25
                        )

                        if res.status_code != 200:
                            ai_msg = f"Backend Error: {res.text}"
                        else:
                            ai_msg = res.json().get("answer", "No answer received.")

                    except Exception as e:
                        ai_msg = f"Request failed: {e}"

                st.markdown(ai_msg)

            # Save AI answer
            st.session_state.chat_history.append(
                {"role": "assistant", "content": ai_msg}
            )

