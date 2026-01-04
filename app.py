import streamlit as st
import pandas as pd
import re
from collections import Counter
import urllib.request
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Shifting Narratives", page_icon="🌍", layout="centered")

#schema
required_columns = ["event_name", "stakeholder", "text", "date"]
optional_columns = [ "source", "collection_method"]

#negation + stopwords
negations = {"not","no","never","without"}
stopwords = {
    "the","and","a","to","of","in","on","for","with","as","at","by",
    "is","are","was","were","be","been","being","it","this","that",
    "from","or","but","we","they","you","i","he","she","them",
    "his","her","their","our","us","your","my","me","will","would",
    "can","could","should","may","might","about","into","over","after",
    "before","more","most","some","any","so","such","also","very","just",
    "up","down","out","now","new"
}

#nrc emotion lexicon
nrc_url = (
    "https://raw.githubusercontent.com/aditeyabaral/"
    "lok-sabha-election-twitter-analysis/master/"
    "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
)

def template_df():
    return pd.DataFrame(
        [{
            "event_name": "example event",
            "stakeholder": "media",
            "text": "the policy was not good and caused a crisis",
            "date": "2026-01-01",
            "source": "example source",
            "collection_method": "manual"
        }],
        columns=required_columns + optional_columns
    )

def validate_csv(df):
    if df is None or df.empty:
        return False, "upload failed: the csv file contains no rows."
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return False, f"upload failed: missing required columns: {missing}"
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if d["date"].isna().any():
        return False, "date format error: please use yyyy-mm-dd in the date column."
    if d["stakeholder"].astype(str).str.strip().eq("").any():
        return False, "validation error: stakeholder column has empty values."
    if d["event_name"].astype(str).str.strip().eq("").any():
        return False, "validation error: event_name column has empty values."
    if d["text"].astype(str).str.strip().eq("").any():
        return False, "validation error: text column has empty values."
    return True, ""

@st.cache_data(show_spinner=False)
def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "streamlit-app"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")

def parse_nrc(raw_text, ignore_words):
    pos = set()
    neg = set()
    for line in raw_text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        word, emotion, value = parts
        if value != "1":
            continue
        word = word.strip().lower()
        if word in ignore_words:
            continue
        if emotion == "positive":
            pos.add(word)
        elif emotion == "negative":
            neg.add(word)
    return pos, neg

@st.cache_data(show_spinner=False)
def load_nrc_cached(ignore_words_tuple):
    raw = fetch_text(nrc_url)
    ignore_words = set(ignore_words_tuple)
    return parse_nrc(raw, ignore_words)

def tokenize(text, ignore_words):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    toks = [t for t in text.split() if len(t) > 2 and t not in stopwords and t not in ignore_words]
    return toks

def sentiment_split_words(text, pos_set, neg_set, ignore_words):
    toks = tokenize(text, ignore_words)
    counts = Counter()
    pos_words = []
    neg_words = []
    neu_words = []

    i = 0
    while i < len(toks):
        w = toks[i]

        if w in negations and i + 1 < len(toks):
            nxt = toks[i + 1]
            if nxt in pos_set:
                counts["negative"] += 1
                neg_words.append(f"{w} {nxt}")
                i += 2
                continue
            if nxt in neg_set:
                counts["positive"] += 1
                pos_words.append(f"{w} {nxt}")
                i += 2
                continue

        if w in pos_set:
            counts["positive"] += 1
            pos_words.append(w)
        elif w in neg_set:
            counts["negative"] += 1
            neg_words.append(w)
        else:
            counts["neutral"] += 1
            neu_words.append(w)

        i += 1

    return counts, pos_words, neg_words, neu_words

def sentiment_score(counts):
    pos = counts["positive"]
    neg = counts["negative"]
    neu = counts["neutral"]
    total = pos + neg + neu
    score = 0.0 if total == 0 else (pos - neg) / total
    label = "positive" if score > 0.08 else "negative" if score < -0.08 else "neutral"
    return score, label, total

def make_freq(words):
    return dict(Counter(words))

def draw_cloud(freqs, color_hex, title):
    if not freqs:
        st.caption(f"{title}: no words")
        return
    wc = WordCloud(width=800, height=400, background_color="white", collocations=False)
    wc = wc.generate_from_frequencies(freqs).recolor(color_func=lambda *args, **kwargs: color_hex)
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.markdown(f"**{title}**")
    st.pyplot(fig, clear_figure=True)

def show_words_as_text(words, title):
    c = Counter(words)
    if not c:
        st.caption(f"{title}: no words")
        return
    dfw = pd.DataFrame(c.most_common(50), columns=["word", "count"])
    st.markdown(f"**{title} (top 50)**")
    st.dataframe(dfw, use_container_width=True, hide_index=True)

def stakeholder_summary(df_subset, pos_set, neg_set, ignore_words):
    combined_text = " ".join(df_subset["text"].astype(str).tolist())
    counts, pos_words, neg_words, neu_words = sentiment_split_words(combined_text, pos_set, neg_set, ignore_words)
    score, label, total = sentiment_score(counts)
    return {
        "score": score,
        "label": label,
        "counts": counts,
        "total": total,
        "pos_words": pos_words,
        "neg_words": neg_words,
        "neu_words": neu_words
    }

#state
if "dataset" not in st.session_state:
    st.session_state.dataset = pd.DataFrame(columns=required_columns + optional_columns)

if "ignore_words" not in st.session_state:
    st.session_state.ignore_words = set()

if "user_pos_words" not in st.session_state:
    st.session_state.user_pos_words = set()

if "user_neg_words" not in st.session_state:
    st.session_state.user_neg_words = set()

#sidebar
st.sidebar.title("Controls")
st.sidebar.markdown(
    """
    **note:** sentiment parsing and keyword collection are based on the  
    **NRC Emotion Lexicon**, a human-annotated sentiment word set.  

    <a href="https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm"
       target="_blank"
       style="color:#e83e8c; font-weight:600; text-decoration:none;">
       learn more about the NRC Emotion Lexicon
    </a>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
input_mode = st.sidebar.radio("Input Method", ["upload csv", "paste text"])
st.sidebar.markdown("---")

if input_mode == "upload csv":
    st.sidebar.subheader("Instructions")
    st.sidebar.markdown(
        "- required columns: `event_name, stakeholder, text, date`\n"
        "- date format: `yyyy-mm-dd`\n"
        "- optional: `source, collection_method`\n"
        "- use the template if unsure"
    )

    st.sidebar.code(template_df().to_csv(index=False))
    st.sidebar.download_button(
        label="download template csv",
        data=template_df().to_csv(index=False).encode("utf-8"),
        file_name="shifting_narratives_template.csv",
        mime="text/csv"
    )

    up = st.sidebar.file_uploader("Upload csv", type=["csv"])
    if up is not None:
        try:
            df_new = pd.read_csv(up)
            ok, msg = validate_csv(df_new)
            if not ok:
                st.sidebar.error(msg)
            else:
                for c in optional_columns:
                    if c not in df_new.columns:
                        df_new[c] = ""
                df_new["date"] = pd.to_datetime(df_new["date"])
                st.session_state.dataset = df_new.copy()
                st.sidebar.success("csv loaded successfully.")
        except Exception:
            st.sidebar.error("upload failed: could not read the csv file. please check formatting.")

else:
    st.sidebar.subheader("instructions")
    st.sidebar.markdown(
        "- fill event name + stakeholder + date\n"
        "- paste article text\n"
        "- click `add` to store it"
    )

    event_name = st.sidebar.text_input("event name")
    stakeholder = st.sidebar.text_input("stakeholder")
    date = st.sidebar.date_input("date")
    with st.sidebar.expander("Optional fields", expanded=False):
        source = st.text_input("source")
        collection_method = st.text_input("collection method")
    text = st.sidebar.text_area("paste article text", height=220)

    if st.sidebar.button("add"):
        if not event_name.strip():
            st.sidebar.error("missing event name.")
        elif not stakeholder.strip():
            st.sidebar.error("missing stakeholder.")
        elif not text.strip():
            st.sidebar.error("missing text.")
        else:
            row = {
                "event_name": event_name.strip(),
                "stakeholder": stakeholder.strip(),
                "text": text.strip(),
                "date": str(date),
                "source": source.strip(),
                "collection_method": collection_method.strip()
            }
            st.session_state.dataset = pd.concat(
                [st.session_state.dataset, pd.DataFrame([row])],
                ignore_index=True
            )
            st.session_state.dataset["date"] = pd.to_datetime(st.session_state.dataset["date"], errors="coerce")
            st.sidebar.success("added.")

#main
st.title("Shifting Narratives")
df = st.session_state.dataset.copy()

if df.empty:
    st.info("no data yet. use the left panel to upload a csv or paste text.")
    st.stop()

for c in required_columns:
    if c not in df.columns:
        df[c] = ""
for c in optional_columns:
    if c not in df.columns:
        df[c] = ""

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()
df["text"] = df["text"].astype(str)

st.sidebar.markdown("---")
st.sidebar.subheader("view")

events = sorted([e for e in df["event_name"].dropna().unique() if str(e).strip() != ""])
if not events:
    st.warning("dataset is missing usable event_name values.")
    st.stop()

selected_event = st.sidebar.selectbox("event", events, index=None)
if selected_event is None:
    st.info("select an event to continue.")
    st.stop()

event_df = df[df["event_name"] == selected_event].copy()
stakeholders_for_event = sorted([s for s in event_df["stakeholder"].dropna().unique() if str(s).strip() != ""])
if not stakeholders_for_event:
    st.warning("no stakeholders found for this event.")
    st.stop()

st.sidebar.markdown("---")
view_kind = st.sidebar.radio("view mode", ["single stakeholder", "compare stakeholders"])

if view_kind == "single stakeholder":
    selected_stake = st.sidebar.selectbox("stakeholder", stakeholders_for_event, index=None)
    if selected_stake is None:
        st.info("select a stakeholder to view.")
        st.stop()
else:
    compare_stakes = st.sidebar.multiselect("stakeholders to compare", stakeholders_for_event, default=[])
    if len(compare_stakes) < 2:
        st.info("select at least 2 stakeholders to compare.")
        st.stop()

view_mode = st.sidebar.radio("Word Display", ["word clouds", "text list"])

st.sidebar.markdown("---")
st.sidebar.subheader("Refine Keyword Set ")
new_ignore = st.sidebar.text_input("add a word to ignore", value="").strip().lower()
add_clicked = st.sidebar.button("add to ignore list")
if add_clicked and new_ignore:
    cleaned = re.sub(r"[^a-z]", "", new_ignore)
    if cleaned:
        st.session_state.ignore_words.add(cleaned)

if st.session_state.ignore_words:
    st.sidebar.caption("ignored words:")
    st.sidebar.write(", ".join(sorted(st.session_state.ignore_words)))
    if st.sidebar.button("clear ignore list"):
        st.session_state.ignore_words = set()

st.sidebar.markdown("---")
st.sidebar.subheader("Adjust Sentiment Interpretation")

pos_in = st.sidebar.text_input("mark a word as positive", value="").strip().lower()
if st.sidebar.button("add positive"):
    cleaned = re.sub(r"[^a-z]", "", pos_in)
    if cleaned:
        st.session_state.user_pos_words.add(cleaned)

neg_in = st.sidebar.text_input("mark a word as negative", value="").strip().lower()
if st.sidebar.button("add negative"):
    cleaned = re.sub(r"[^a-z]", "", neg_in)
    if cleaned:
        st.session_state.user_neg_words.add(cleaned)

if st.session_state.user_pos_words:
    st.sidebar.caption("custom positive:")
    st.sidebar.write(", ".join(sorted(st.session_state.user_pos_words)))
if st.session_state.user_neg_words:
    st.sidebar.caption("custom negative:")
    st.sidebar.write(", ".join(sorted(st.session_state.user_neg_words)))

if st.sidebar.button("Reset"):
    st.session_state.user_pos_words = set()
    st.session_state.user_neg_words = set()

ignore_words = set(st.session_state.ignore_words)

#load base lexicon then apply overrides
lex_err = None
pos_set, neg_set = set(), set()
try:
    pos_set, neg_set = load_nrc_cached(tuple(sorted(ignore_words)))
    pos_set = (pos_set | set(st.session_state.user_pos_words)) - ignore_words
    neg_set = (neg_set | set(st.session_state.user_neg_words)) - ignore_words
except Exception as e:
    lex_err = str(e)

if lex_err is not None:
    st.error("sentiment word set is not loaded.")
    st.stop()

if view_kind == "single stakeholder":
    view = event_df[event_df["stakeholder"] == selected_stake].copy().sort_values("date")

    if view.empty:
        st.warning("no documents match the selected filters.")
        st.stop()

    res = stakeholder_summary(view, pos_set, neg_set, ignore_words)
    counts = res["counts"]
    score = res["score"]
    label = res["label"]
    total = res["total"]

    st.subheader(f"{selected_stake} — {selected_event}")
    st.markdown(f"**sentiment score:** `{score:.2f}` (**{label}**)")

    m1, m2, m3 = st.columns(3)
    m1.metric("positive", counts["positive"])
    m2.metric("negative", counts["negative"])
    m3.metric("neutral", counts["neutral"])

    st.markdown("### how the score is calculated")
    st.markdown(
        f"- positive count = `{counts['positive']}`\n"
        f"- negative count = `{counts['negative']}`\n"
        f"- neutral count = `{counts['neutral']}`\n"
        f"- total = `{total}`\n"
        f"- score = (positive - negative) / total = "
        f"({counts['positive']} - {counts['negative']}) / {total} = `{score:.4f}`\n"
        f"- label rule: > 0.08 = positive, < -0.08 = negative, else neutral"
    )

    st.markdown("### words used to compute the score")
    if view_mode == "word clouds":
        cpos, cneg, cneu = st.columns(3)
        with cpos:
            draw_cloud(make_freq(res["pos_words"]), "#00aa00", "positive")
        with cneg:
            draw_cloud(make_freq(res["neg_words"]), "#cc0000", "negative")
        with cneu:
            draw_cloud(make_freq(res["neu_words"]), "#888888", "neutral")
    else:
        cpos, cneg, cneu = st.columns(3)
        with cpos:
            show_words_as_text(res["pos_words"], "positive")
        with cneg:
            show_words_as_text(res["neg_words"], "negative")
        with cneu:
            show_words_as_text(res["neu_words"], "neutral")

else:
    st.subheader(f"comparison — {selected_event}")

    summaries = []
    per_stake = {}

    for s in compare_stakes:
        sub = event_df[event_df["stakeholder"] == s].copy()
        if sub.empty:
            continue

        res = stakeholder_summary(sub, pos_set, neg_set, ignore_words)
        per_stake[s] = res
        c = res["counts"]

        summaries.append({
            "stakeholder": s,
            "score": res["score"],
            "label": res["label"],
            "positive": c["positive"],
            "negative": c["negative"],
            "neutral": c["neutral"],
            "total": res["total"],
            "pos_minus_neg": c["positive"] - c["negative"],
            "documents": len(sub)
        })

    if not summaries:
        st.warning("no data available for the selected stakeholders")
        st.stop()

    df_sum = pd.DataFrame(summaries).sort_values("score", ascending=False)

    st.markdown("### scores")
    card_cols = st.columns(min(4, len(df_sum)))
    for i, row in enumerate(df_sum.itertuples(index=False)):
        col = card_cols[i % len(card_cols)]
        with col:
            st.metric(
                f"{row.stakeholder}",
                f"{row.score:.2f}",
                help=f"label: {row.label} | +{row.positive} / -{row.negative} / n{row.neutral}"
            )
            st.caption(f"label: {row.label}")
            st.caption(f"+{row.positive} / -{row.negative} / n{row.neutral}")
            st.caption(f"docs: {row.documents}")

    st.markdown("### positive vs negative (counts)")
    plot_df = df_sum.copy()
    plot_df["neg_left"] = -plot_df["negative"]
    plot_df = plot_df.sort_values("score", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(2.8, 0.6 * len(plot_df))))
    ax.barh(plot_df["stakeholder"], plot_df["neg_left"])
    ax.barh(plot_df["stakeholder"], plot_df["positive"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("count (negative left, positive right)")
    ax.set_ylabel("")
    st.pyplot(fig, clear_figure=True)

    st.markdown("### words used (per stakeholder)")
    for s in plot_df["stakeholder"].tolist():
        res = per_stake[s]
        st.markdown("---")
        st.subheader(s)
        st.markdown(f"**score:** `{res['score']:.2f}` (**{res['label']}**)")

        c = res["counts"]
        m1, m2, m3 = st.columns(3)
        m1.metric("positive", c["positive"])
        m2.metric("negative", c["negative"])
        m3.metric("neutral", c["neutral"])

        if view_mode == "word clouds":
            cpos, cneg, cneu = st.columns(3)
            with cpos:
                draw_cloud(make_freq(res["pos_words"]), "#00aa00", "positive")
            with cneg:
                draw_cloud(make_freq(res["neg_words"]), "#cc0000", "negative")
            with cneu:
                draw_cloud(make_freq(res["neu_words"]), "#888888", "neutral")
        else:
            cpos, cneg, cneu = st.columns(3)
            with cpos:
                show_words_as_text(res["pos_words"], "positive")
            with cneg:
                show_words_as_text(res["neg_words"], "negative")
            with cneu:
                show_words_as_text(res["neu_words"], "neutral")
