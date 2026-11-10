import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# ---------------------------------------------------------------
# Lesson 1: Sample data and sentiment classification
# ---------------------------------------------------------------
reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "Good value, but cleaning the narrow opening is difficult.",
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality.",
]

df = pd.DataFrame({"id": [f"R{i:02}" for i in range(1, 19)],
                   "text": reviews})

analyzer = SentimentIntensityAnalyzer()


def label_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    if score <= -0.05:
        return "Negative", score
    return "Neutral", score


df[["sentiment", "compound"]] = df["text"].apply(
    lambda text: pd.Series(label_sentiment(text)))

# Task 4: main topic for each comment (one per row, 18 in total)
topics = [
    "insulation",  # R01
    "size",        # R02
    "design",      # R03
    "lid",         # R04
    "lid",         # R05
    "cleaning",    # R06
    "insulation",  # R07
    "general",     # R08
    "insulation",  # R09
    "cleaning",    # R10
    "capacity",    # R11
    "general",     # R12
    "lid",         # R13
    "delivery",    # R14
    "insulation",  # R15
    "price",       # R16
    "handle",      # R17
    "quality",     # R18
]
df["topic"] = topics

# ---------------------------------------------------------------
# Lesson 2: Dashboard
# ---------------------------------------------------------------
st.set_page_config(page_title="Lab 8 | Feedback", layout="wide")
st.title("Product Feedback: A Text Summary")
st.caption("Practice data only. Automated labels need human review.")

left, right = st.columns(2)

with left:
    st.subheader("Comment counts by sentiment")
    order = ["Positive", "Neutral", "Negative"]
    counts = df["sentiment"].value_counts().reindex(order, fill_value=0)
    st.bar_chart(counts, horizontal=True)
    st.caption("Each bar counts comments by automated label. "
               "A comment with mixed opinions still gets only one label.")

with right:
    st.subheader("Words used in the comments")
    # Task 2: "drinks" was REMOVED from the custom stop list so it is retained.
    # "keeps" was ADDED because it is a frequent filler verb that tells readers little.
    stop_words = set(STOPWORDS) | {"bottle", "drink", "keeps"}
    cloud = WordCloud(width=900, height=500, background_color="white",
                      colormap="viridis", stopwords=stop_words,
                      collocations=False).generate(" ".join(df["text"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    st.caption("Larger words appear more often in the comments. Size shows "
               "frequency only, not sentiment. Position and color carry no "
               "meaning, and with only 18 comments a few repeats can dominate.")

st.subheader("Review the original comments")
st.dataframe(df[["id", "text", "topic", "sentiment", "compound"]],
             use_container_width=True, hide_index=True)
st.caption("A compound score is a model output, not a confidence rating.")
st.write(f"Total comments displayed: {len(df)}")

# ---------------------------------------------------------------
# Task 3: Audit sentiment labels
# ---------------------------------------------------------------
st.subheader("Audit: manual review of six labels")
st.caption("Fill in the 'reasonable' and 'note' values after reading each comment.")

# EDIT the ids below to the six comments you choose (2 positive, 2 neutral, 2 negative)
# and update "reasonable" and "note" with your own judgement.
audit_ids = ["R02", "R12", "R01", "R14", "R04", "R13"]
audit = df[df["id"].isin(audit_ids)][["id", "text", "sentiment", "compound"]].copy()
audit_review = {
    "R01": ("No", "Clearly positive (keeps water cold all day), but it has no strong sentiment words, so VADER scored it 0 and called it Neutral."),
    "R02": ("Yes", "Words like 'great' and 'easy' make this clearly positive."),
    "R04": ("Yes", "'Leaks' and 'very disappointing' are clearly negative."),
    "R12": ("Yes", "'Excellent' and daily use make this clearly positive."),
    "R13": ("Yes", "'Broke' and 'poor quality' are clearly negative."),
    "R14": ("Yes (borderline)", "A factual statement, so Neutral is reasonable, though 'works as described' is mildly positive."),
}
audit["reasonable"] = audit["id"].map(lambda i: audit_review[i][0])
audit["note"] = audit["id"].map(lambda i: audit_review[i][1])
st.dataframe(audit, use_container_width=True, hide_index=True)

st.subheader("Negation test")
test = "The bottle is not easy to clean."
scores = analyzer.polarity_scores(test)
label, _ = label_sentiment(test)
st.write(f'Test sentence: "{test}"')
st.write(f"VADER label: **{label}**")
st.json(scores)

# ---------------------------------------------------------------
# Task 4: Sentiment by topic
# ---------------------------------------------------------------
st.subheader("Sentiment by topic")
topic_table = pd.crosstab(df["topic"], df["sentiment"]).reindex(
    columns=order, fill_value=0)
topic_table["total"] = topic_table.sum(axis=1)
st.dataframe(topic_table.sort_values("total", ascending=False),
             use_container_width=True)
st.caption("Most topics have only one to four comments, so this sample is too "
           "small for a fair comparison between topics.")

# ---------------------------------------------------------------
# Task 4: Summary for a product team (replace with your own words, 150-200)
# ---------------------------------------------------------------
st.subheader("Summary for the product team")
st.write("""
Of the 18 practise comments, VADER labeled 9 positive, 6 negative and 3 neutral, so positive is the largest group. Customers praised
ease of carrying and washing, and overall quality. The clearest problem was the lid: all three lid comments were labeled negative, 
citing leaking, a hard-to-open cap and breakage. 

This finding has limits. With only 18 synthetic comments, no topic has enough responses for a fair comparison. 
VADER also misreads context. "The bottle keeps my water cold all day" was labeled neutral, and two insulation complaints ("I expected better insulation for this price" and "it does not
keep drinks cold") were labeled positive, probably because of words like "better" and "nice". Mixed comments, such as "Good value, but cleaning the narrow opening is difficult", receive one label that hides 
either the praise or the complaint. As a result, the counts may overstate satisfaction. I recommend gathering more feedbacks by pairing each comment with a star rating and a topic-specific question, such as "How well did the lid work?". 
A person should also manually check a random sample of automated labels before the team acts on them)
""")