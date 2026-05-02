import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Scam & Phishing RAG Dashboard",
    page_icon="🛡️",
    layout="wide"
)

LOG_PATH = Path("logs/chat_history.jsonl")


def load_logs():
    rows = []

    if not LOG_PATH.exists():
        return pd.DataFrame()

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception:
                continue

            rows.append(data)

    return pd.DataFrame(rows)


def normalize_df(df):
    if df.empty:
        return df

    # timestamp
    if "timestamp" not in df.columns:
        df["timestamp"] = datetime.now().isoformat()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # message / question
    if "user_message" not in df.columns:
        if "question" in df.columns:
            df["user_message"] = df["question"]
        elif "message" in df.columns:
            df["user_message"] = df["message"]
        else:
            df["user_message"] = ""

    # answer
    if "answer" not in df.columns:
        if "bot_response" in df.columns:
            df["answer"] = df["bot_response"]
        elif "response" in df.columns:
            df["answer"] = df["response"]
        else:
            df["answer"] = ""

    # label / classification
    if "label" not in df.columns:
        if "classification" in df.columns:
            df["label"] = df["classification"]
        elif "prediction" in df.columns:
            df["label"] = df["prediction"]
        elif "risk_label" in df.columns:
            df["label"] = df["risk_label"]
        else:
            df["label"] = "Unknown"

    # risk score
    if "risk_score" not in df.columns:
        if "score" in df.columns:
            df["risk_score"] = df["score"]
        elif "confidence" in df.columns:
            df["risk_score"] = df["confidence"]
        else:
            df["risk_score"] = 0

    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0)

    # latency
    if "latency" not in df.columns:
        if "latency_seconds" in df.columns:
            df["latency"] = df["latency_seconds"]
        elif "response_time" in df.columns:
            df["latency"] = df["response_time"]
        else:
            df["latency"] = 0

    df["latency"] = pd.to_numeric(df["latency"], errors="coerce").fillna(0)

    # mode
    if "mode" not in df.columns:
        df["mode"] = "Unknown"

    # file type
    if "file_type" not in df.columns:
        df["file_type"] = "text"

    # error
    if "error" not in df.columns:
        df["error"] = ""

    return df


df = normalize_df(load_logs())

st.title("🛡️ Scam & Phishing RAG Chatbot Dashboard")
st.caption("Monitoring dashboard for requests, phishing alerts, suspicious files, errors, latency, and high-risk events.")

if df.empty:
    st.warning("No logs found yet. تأكدي أن الملف موجود هنا: logs/chat_history.jsonl")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

labels = ["All"] + sorted(df["label"].dropna().astype(str).unique().tolist())
selected_label = st.sidebar.selectbox("Risk Label", labels)

modes = ["All"] + sorted(df["mode"].dropna().astype(str).unique().tolist())
selected_mode = st.sidebar.selectbox("Mode", modes)

file_types = ["All"] + sorted(df["file_type"].dropna().astype(str).unique().tolist())
selected_file_type = st.sidebar.selectbox("File Type", file_types)

min_score = st.sidebar.slider("Minimum Risk Score", 0, 100, 0)

filtered = df.copy()

if selected_label != "All":
    filtered = filtered[filtered["label"].astype(str) == selected_label]

if selected_mode != "All":
    filtered = filtered[filtered["mode"].astype(str) == selected_mode]

if selected_file_type != "All":
    filtered = filtered[filtered["file_type"].astype(str) == selected_file_type]

filtered = filtered[filtered["risk_score"] >= min_score]

# Metrics
total_requests = len(filtered)

phishing_alerts = filtered[
    filtered["label"].astype(str).str.lower().str.contains("phishing|scam|malicious", na=False)
].shape[0]

suspicious_files = filtered[
    (filtered["file_type"].astype(str).str.lower() != "text") &
    (filtered["label"].astype(str).str.lower().str.contains("suspicious|phishing|scam|malicious", na=False))
].shape[0]

errors = filtered[
    filtered["error"].astype(str).str.strip() != ""
].shape[0]

avg_latency = round(filtered["latency"].mean(), 2) if total_requests > 0 else 0

high_risk_events = filtered[
    filtered["risk_score"] >= 80
].shape[0]

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric("Total Requests", total_requests)
col2.metric("Phishing Alerts", phishing_alerts)
col3.metric("Suspicious Files", suspicious_files)

col4.metric("Errors", errors)
col5.metric("Average Latency", f"{avg_latency} sec")
col6.metric("High Risk Events", high_risk_events)

st.divider()

# Charts
left, right = st.columns(2)

with left:
    st.subheader("Risk Label Distribution")
    label_counts = filtered["label"].astype(str).value_counts()
    st.bar_chart(label_counts)

with right:
    st.subheader("Mode Distribution")
    mode_counts = filtered["mode"].astype(str).value_counts()
    st.bar_chart(mode_counts)

st.subheader("Latency Over Time")
latency_df = filtered.dropna(subset=["timestamp"]).sort_values("timestamp")
if not latency_df.empty:
    st.line_chart(latency_df.set_index("timestamp")["latency"])
else:
    st.info("No timestamp data available for latency chart.")

st.subheader("Risk Score Over Time")
risk_df = filtered.dropna(subset=["timestamp"]).sort_values("timestamp")
if not risk_df.empty:
    st.line_chart(risk_df.set_index("timestamp")["risk_score"])
else:
    st.info("No timestamp data available for risk score chart.")

st.divider()

# High risk table
st.subheader("High Risk Events")
high_risk_df = filtered[filtered["risk_score"] >= 80].copy()

if high_risk_df.empty:
    st.success("No high-risk events found.")
else:
    show_cols = [
        "timestamp",
        "user_message",
        "label",
        "risk_score",
        "mode",
        "file_type",
        "latency"
    ]
    show_cols = [c for c in show_cols if c in high_risk_df.columns]
    st.dataframe(high_risk_df[show_cols].sort_values("timestamp", ascending=False), use_container_width=True)

# Full logs
st.subheader("Full Chat Logs")

show_cols = [
    "timestamp",
    "user_message",
    "answer",
    "label",
    "risk_score",
    "mode",
    "file_type",
    "latency",
    "error"
]
show_cols = [c for c in show_cols if c in filtered.columns]

st.dataframe(
    filtered[show_cols].sort_values("timestamp", ascending=False),
    use_container_width=True
)

# Download
csv = filtered[show_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Logs as CSV",
    data=csv,
    file_name="filtered_chat_logs.csv",
    mime="text/csv"
)