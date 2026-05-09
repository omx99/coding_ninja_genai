# ===============================================================
# STEP 8 — Final app (everything wired together)
# ---------------------------------------------------------------
# New: API status badge, polished header, CSV download button,
# and a small caption with elapsed time.
# This file is equivalent to ../streamlit.py.
# Run:  streamlit run steps/step8_full.py
# ===============================================================

import time
import requests
import pandas as pd
import streamlit as st

API_BASE = "http://localhost:8000"
BATCH_URL = f"{API_BASE}/batch/"
VERSIONS_URL = f"{API_BASE}/version"
PROMPT_VERSIONS = ["default", "v1_zero_shot", "v2_few_shot", "v3_cot"]

st.set_page_config(page_title="Scam Guard", page_icon="🛡️", layout="wide")


@st.cache_data(ttl=15, show_spinner=False)
def fetch_versions():
    r = requests.get(VERSIONS_URL, timeout=5)
    r.raise_for_status()
    return r.json()


def call_batch(file_bytes, file_name, prompt_version, sample_size):
    files = {"file": (file_name, file_bytes, "text/csv")} if file_bytes else None
    data = {}
    if prompt_version and prompt_version != "default":
        data["prompt_version"] = prompt_version
    if sample_size:
        data["sample_size"] = str(int(sample_size))
    r = requests.post(BATCH_URL, files=files, data=data, timeout=600)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"API error {r.status_code}: {detail}")
    return r.json()


# Header + API status badge
left, right = st.columns([4, 1])
with left:
    st.title("🛡️ Scam Guard")
    st.caption("Step 8: the full dashboard.")
with right:
    try:
        v = fetch_versions()
        st.success("API: online")
        st.caption(f"Default: `{v.get('default_version', '—')}`")
    except Exception:
        st.error("API: offline")

st.divider()

# Sidebar controls
with st.sidebar:
    st.header("Run analysis")
    prompt_version = st.selectbox("Prompt version", PROMPT_VERSIONS, index=0)
    sample_size = st.number_input("Sample size", min_value=0, max_value=10000, value=20, step=5)
    uploaded = st.file_uploader("CSV file (optional)", type=["csv"])
    run = st.button("Run analysis", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None

if run:
    file_bytes = uploaded.read() if uploaded else None
    file_name = uploaded.name if uploaded else None
    with st.spinner("Calling /batch/ ..."):
        t0 = time.time()
        try:
            st.session_state.result = call_batch(
                file_bytes, file_name, prompt_version,
                sample_size if sample_size > 0 else None,
            )
            st.session_state.elapsed = time.time() - t0
        except Exception as e:
            st.session_state.result = None
            st.error(str(e))

result = st.session_state.result
if not result:
    st.info("Configure options on the left and click **Run analysis**.")
    st.stop()

summary = result["summary"]
df = pd.DataFrame(result["results"])

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total processed", summary["total_processed"])
c2.metric("Accuracy", f"{summary['accuracy'] * 100:.1f}%")
c3.metric("Scams detected", summary["scam_detected"])
c4.metric(
    "Not scam / uncertain",
    summary["not_scam_detected"] + summary["uncertain_detected"],
)
st.caption(
    f"Prompt version used: `{summary['prompt_version_used']}` · "
    f"correct {summary['correct_predictions']}/{summary['total_processed']}"
    + (f" · took {st.session_state.get('elapsed', 0):.1f}s" if st.session_state.get("elapsed") else "")
)

st.divider()

# Charts
g1, g2 = st.columns(2)
with g1:
    st.subheader("Label distribution")
    label_counts = df["predicted_label"].value_counts().rename_axis("label").reset_index(name="count")
    st.bar_chart(label_counts, x="label", y="count")
with g2:
    st.subheader("Intent breakdown")
    intent_counts = df["intent_type"].value_counts().rename_axis("intent").reset_index(name="count")
    st.bar_chart(intent_counts, x="intent", y="count", horizontal=True)

st.divider()

# Filters + table
st.subheader("Results")
f1, f2, f3 = st.columns(3)
with f1:
    label_filter = st.multiselect(
        "Predicted label",
        sorted(df["predicted_label"].unique()),
        default=sorted(df["predicted_label"].unique()),
    )
with f2:
    correctness = st.selectbox("Correctness", ["All", "Correct only", "Incorrect only"])
with f3:
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

view = df[df["predicted_label"].isin(label_filter) & (df["confidence_score"] >= min_conf)]
if correctness == "Correct only":
    view = view[view["is_correct"]]
elif correctness == "Incorrect only":
    view = view[~view["is_correct"]]

st.dataframe(
    view[["message", "actual_label", "predicted_label", "is_correct",
          "intent_type", "confidence_score", "reasoning"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "confidence_score": st.column_config.ProgressColumn(
            "confidence", min_value=0.0, max_value=1.0, format="%.2f"
        ),
        "is_correct": st.column_config.CheckboxColumn("correct"),
        "message": st.column_config.TextColumn("message", width="large"),
        "reasoning": st.column_config.TextColumn("reasoning", width="large"),
    },
)

# Download filtered view
st.download_button(
    "Download results as CSV",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="scam_guard_results.csv",
    mime="text/csv",
)
