from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from google import genai
from dotenv import load_dotenv
import pandas as pd
from sklearn.ensemble import IsolationForest
import os, io

load_dotenv()

app = FastAPI()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# --------------------------
# ✅ Severity Scoring Function
# --------------------------
def classify_severity(score, residual, score_thresholds):
    if score > score_thresholds["critical"] or abs(residual) > 2 * score_thresholds["std_residual"]:
        return "Critical"
    elif score > score_thresholds["high"]:
        return "High"
    elif score > score_thresholds["medium"]:
        return "Medium"
    else:
        return "Low"


# --------------------------
# ✅ Root Cause Ranking (Fixed)
# --------------------------
def rank_root_causes(row, metric_col):
    causes = []

    actual_value = row[metric_col]

    # Missing or zero data
    if actual_value == 0:
        causes.append(("Data Pipeline Failure", 0.95))
        causes.append(("Source System Outage", 0.85))
        causes.append(("Manual Entry Missing", 0.75))

    # Large positive spike
    elif row["residual"] > abs(row["rolling_mean"]) * 0.8:
        causes.append(("Duplicate Transaction", 0.92))
        causes.append(("Large One-time Transaction", 0.88))
        causes.append(("Data Entry Error", 0.80))

    # Large negative dip
    elif row["residual"] < -abs(row["rolling_mean"]) * 0.4:
        causes.append(("Missing Partial Data", 0.90))
        causes.append(("Reporting System Delay", 0.82))
        causes.append(("Sudden Operational Drop", 0.70))

    return sorted(causes, key=lambda x: x[1], reverse=True)


# --------------------------
# ✅ Anomaly Detection
# --------------------------
def detect_anomalies(df, time_col, metric_col):
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    df["rolling_mean"] = df[metric_col].rolling(window=5, min_periods=1).mean()
    df["residual"] = df[metric_col] - df["rolling_mean"]

    df["zscore"] = (df["residual"] - df["residual"].mean()) / (df["residual"].std() + 1e-9)

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(df[["residual", "zscore"]])

    df["score"] = -model.decision_function(df[["residual", "zscore"]])

    score_thresholds = {
        "medium": df["score"].quantile(0.75),
        "high": df["score"].quantile(0.90),
        "critical": df["score"].quantile(0.97),
        "std_residual": df["residual"].std()
    }

    df["severity"] = df.apply(
        lambda row: classify_severity(row["score"], row["residual"], score_thresholds),
        axis=1
    )

    df["is_anomaly"] = df["severity"] != "Low"

    return df[df["is_anomaly"]]


# --------------------------
# ✅ AI Explanation
# --------------------------
def explain_anomaly(row, time_col, metric_col, ranked_causes):

    causes_str = "\n".join([f"- {c[0]} (confidence: {int(c[1]*100)}%)" for c in ranked_causes])

    prompt = f"""
Explain this financial anomaly clearly and concisely:

Timestamp: {row[time_col]}
Actual Value: {row[metric_col]}
Rolling Mean Expected: {row['rolling_mean']}
Residual: {row['residual']}
Severity: {row['severity']}

Ranked Root Causes:
{causes_str}

Your structured response must include:

=================================================
### ✅ 1. EXPLANATION (Max 4–5 lines)
- What this anomaly means  
- Why it likely happened  
- Why it matters  
- Business risk level  

=================================================
### ✅ 2. ACTIONABLE RECOMMENDATIONS (Step-by-step)
**Immediate fixes (today)**  
- 3–5 specific steps  

**Team Responsible**  
- Finance / Ops / Data Engineering  

**Business Impact (1 line)**  

**Prevention Steps**  
- 3 long-term prevention measures  
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return resp.text


# --------------------------
# ✅ API Endpoint
# --------------------------
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    time_col: str = Form(...),
    metric_col: str = Form(...)
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        anomalies = detect_anomalies(df, time_col, metric_col)

        results = []
        for _, row in anomalies.iterrows():

            ranked_causes = rank_root_causes(row, metric_col)

            results.append({
                "timestamp": str(row[time_col]),
                "value": float(row[metric_col]),
                "rolling_mean": float(row["rolling_mean"]),
                "residual": float(row["residual"]),
                "score": float(row["score"]),
                "severity": row["severity"],
                "root_causes_ranked": ranked_causes,
                "explanation": explain_anomaly(row, time_col, metric_col, ranked_causes)
            })

        return {"anomalies": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/chat")
async def chat(payload: dict):
    try:
        question = payload.get("question", "")
        dataset_summary = payload.get("dataset_summary", "")

        prompt = f"""
You are a data analyst AI.

Dataset context:
{dataset_summary}

User question:
{question}

Give a clear, business-friendly answer.
"""

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return {"answer": resp.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

