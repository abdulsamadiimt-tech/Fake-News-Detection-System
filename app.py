"""Streamlit web app.  Run:  streamlit run app.py"""
import os
import re
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from textclean import clean_text, style_features

st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="wide")

st.markdown(
    """
    <style>
    .hero {padding: 1.2rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
           background: linear-gradient(90deg, #1e3a8a, #7c3aed); color: white;}
    .hero h1 {margin: 0; padding: 0; font-size: 2rem; color: white;}
    .hero p {margin: 0.3rem 0 0 0; opacity: 0.9;}
    </style>
    <div class="hero">
      <h1>🕵️ Fake News Detection System</h1>
      <p>Analyze the writing patterns of news text, one post or hundreds at a time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.path.exists("model.joblib"):
    st.error("🚫 Model not found. Run `python train.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


bundle = load_model()
vec, model = bundle["vec"], bundle["model"]
names = vec.get_feature_names_out()

MAX_ITEMS = 2000
ORDER = ["Likely REAL", "Likely FAKE", "Uncertain", "Too short to judge"]
ICON = {"Likely REAL": "✅", "Likely FAKE": "🚫", "Uncertain": "❓", "Too short to judge": "✏️"}
COLORS = {"Likely REAL": "#16a34a", "Likely FAKE": "#dc2626",
          "Uncertain": "#f59e0b", "Too short to judge": "#9ca3af"}

REAL_SAMPLE = ("The U.S. Senate voted on Tuesday to approve a spending bill that would fund federal "
               "agencies through the end of the fiscal year, officials said, sending the measure "
               "to the president for his signature.")
FAKE_SAMPLE = ("SHOCKING!!! You won't BELIEVE what the media is hiding from you about the election. "
               "Share this before they DELETE it!!!")
MULTI_SAMPLE = (
    "The central bank announced on Monday that interest rates would remain unchanged after a two-day meeting.\n"
    "SHOCKING!!! Doctors HATE this one trick and the government is hiding it from you, share before they DELETE it!!!\n"
    "Local council approves new budget for road repairs after a public hearing on Thursday."
)

# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    real_cut = st.slider("✅ Likely REAL if probability is at least", 0.50, 0.95, 0.65, 0.05)
    fake_cut = st.slider("🚫 Likely FAKE if probability is at most", 0.05, 0.50, 0.35, 0.05)
    min_words = st.slider("✏️ Minimum words to judge", 3, 30, 8)
    st.caption("Anything between the two limits is shown as ❓ Uncertain.")
    st.divider()
    st.subheader("💡 Tips")
    st.markdown("- 📄 Use full sentences, not bare headlines\n"
                "- 📋 Use **Check many posts** for lists\n"
                "- 🔎 Always confirm with trusted sources")
    st.info("🧠 This tool judges **writing style**, not facts. It can be wrong.")


# ---------------- Helper functions ----------------
def verdict(p_real, n_words):
    if n_words < min_words:
        return "Too short to judge"
    if p_real >= real_cut:
        return "Likely REAL"
    if p_real <= fake_cut:
        return "Likely FAKE"
    return "Uncertain"


def analyze(texts):
    texts = [str(t) for t in texts][:MAX_ITEMS]
    cleaned = [clean_text(t) for t in texts]
    probs = model.predict_proba(vec.transform(cleaned))[:, 1]
    rows = []
    for i, (t, p) in enumerate(zip(texts, probs), start=1):
        n = len(t.split())
        rows.append({
            "#": i,
            "text_preview": (t[:90] + "...") if len(t) > 90 else t,
            "words": n,
            "real_probability": round(float(p), 3),
            "verdict": verdict(p, n),
        })
    return pd.DataFrame(rows)


def banner(label):
    msg = f"{ICON[label]}  {label}"
    if label == "Likely REAL":
        st.success(msg)
    elif label == "Likely FAKE":
        st.error(msg)
    elif label == "Uncertain":
        st.warning(msg + " - the writing style is mixed")
    else:
        st.info(msg + " - add more text for a better result")


def show_results(result, key):
    counts = result["verdict"].value_counts()
    cols = st.columns(4)
    for col, name in zip(cols, ORDER):
        col.metric(f"{ICON[name]} {name}", int(counts.get(name, 0)))

    left, right = st.columns([1, 2])
    with left:
        present = [n for n in ORDER if counts.get(n, 0) > 0]
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.pie([counts[n] for n in present], labels=present,
               colors=[COLORS[n] for n in present], autopct="%1.0f%%",
               startangle=90, wedgeprops={"width": 0.45})
        ax.set_title("Share of each verdict")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        picked = st.multiselect("🔎 Filter by verdict", ORDER, default=ORDER, key=f"{key}_filter")
        shown = result[result["verdict"].isin(picked)]
        display = shown.copy()
        display["verdict"] = display["verdict"].map(lambda v: f"{ICON[v]} {v}")
        st.dataframe(
            display, hide_index=True,
            column_config={"real_probability": st.column_config.ProgressColumn(
                "real_probability", min_value=0.0, max_value=1.0, format="%.2f")},
        )
        st.download_button("⬇️ Download results (CSV)", shown.to_csv(index=False),
                           "results.csv", key=f"{key}_download")


def split_items(raw):
    """One item per line, or articles separated by a line containing only ---."""
    divider = r"^\s*-{3,}\s*$"
    if re.search(divider, raw, flags=re.M):
        parts = re.split(divider, raw, flags=re.M)
    else:
        parts = raw.splitlines()
    return [p.strip() for p in parts if p.strip()]


def set_text(key, value):
    st.session_state[key] = value


# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Check one post", "📋 Check many posts", "📁 Upload a CSV", "📊 Model charts"])

with tab1:
    b1, b2, b3, _ = st.columns([1.3, 1.3, 1, 2])
    b1.button("📰 News-style sample", on_click=set_text, args=("single_text", REAL_SAMPLE))
    b2.button("🚨 Sensational sample", on_click=set_text, args=("single_text", FAKE_SAMPLE))
    b3.button("🧹 Clear", on_click=set_text, args=("single_text", ""))

    text = st.text_area("✍️ Paste a headline or article", height=180, key="single_text")

    if st.button("🔍 Analyze", type="primary"):
        if not text.strip():
            st.warning("⚠️ Please paste some text first.")
        else:
            with st.spinner("🧠 Reading the text patterns..."):
                X = vec.transform([clean_text(text)])
                p_real = float(model.predict_proba(X)[0][1])
            banner(verdict(p_real, len(text.split())))
            st.progress(p_real, text=f"Real-news probability: {p_real:.0%}")

            feats = style_features(text)
            m1, m2, m3 = st.columns(3)
            m1.metric("📝 Words", feats["word_count"])
            m2.metric("❗ Exclamation marks", feats["exclamations"])
            m3.metric("🔠 Capital letters", f"{feats['caps_ratio']:.0%}")

            with st.expander("🔬 Why did it decide this? (words that mattered)", expanded=True):
                contrib = X.toarray()[0] * model.coef_[0]
                pairs = sorted(((names[i], contrib[i]) for i in X.nonzero()[1]), key=lambda p: p[1])
                fake_words = [w for w, c in pairs[:5] if c < 0]
                real_words = [w for w, c in pairs[-5:][::-1] if c > 0]
                c1, c2 = st.columns(2)
                c1.markdown("**🚫 Pushing toward FAKE**")
                c1.markdown("\n".join(f"- 🔴 {w}" for w in fake_words) or "_none_")
                c2.markdown("**✅ Pushing toward REAL**")
                c2.markdown("\n".join(f"- 🟢 {w}" for w in real_words) or "_none_")

with tab2:
    st.write("Paste several news items. Put **one item per line**. For long articles with several "
             "paragraphs, put a line containing only `---` between articles.")
    st.button("🧪 Load an example list", on_click=set_text, args=("multi_text", MULTI_SAMPLE))
    raw = st.text_area("📋 News items", height=250, key="multi_text")
    if st.button("🚀 Analyze all", type="primary"):
        if not raw.strip():
            st.warning("⚠️ Please paste some news items first.")
        else:
            items = split_items(raw)
            if len(items) > MAX_ITEMS:
                st.warning(f"Only the first {MAX_ITEMS} items are analyzed.")
            with st.spinner(f"🧠 Analyzing {min(len(items), MAX_ITEMS)} items..."):
                st.session_state["multi_result"] = analyze(items)
            st.toast("Analysis complete!", icon="✅")
    if "multi_result" in st.session_state:
        show_results(st.session_state["multi_result"], key="multi")

with tab3:
    uploaded = st.file_uploader("📁 Upload a CSV file", type="csv")
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            lines = uploaded.read().decode("utf-8", errors="ignore").splitlines()
            lines = [l for l in lines if l.strip()]
            if lines and lines[0].strip().lower() == "text":
                lines = lines[1:]
            data = pd.DataFrame({"text": lines})
            st.warning("The CSV had commas inside the text, so each line was read as one news item.")
        column = st.selectbox("🗂️ Which column has the text?", data.columns)
        st.caption(f"{len(data)} rows found.")
        if st.button("🚀 Run analysis", type="primary"):
            with st.spinner("🧠 Analyzing..."):
                st.session_state["csv_result"] = analyze(data[column].astype(str).tolist())
            st.toast("Analysis complete!", icon="✅")
    if "csv_result" in st.session_state:
        show_results(st.session_state["csv_result"], key="csv")

with tab4:
    charts = [("📈 Writing-style patterns", "charts/style_patterns.png"),
              ("🎯 Confusion matrix", "charts/confusion_matrix.png"),
              ("🔤 Most telling words", "charts/top_words.png")]
    for title, path in charts:
        if os.path.exists(path):
            with st.expander(title, expanded=True):
                st.image(path)

st.divider()
st.caption("🎓 College project - Fake News Detection System. Built with Python, pandas, NLTK, "
           "scikit-learn, matplotlib and Streamlit.")
