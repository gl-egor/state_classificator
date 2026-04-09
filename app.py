import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_model():
    name = "glegor/arxiv-classifier"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    return tok, model

tok, model = load_model()
id2label = model.config.id2label

st.title("Классификатор статей arxiv")

title = st.text_input("Title", key="title_input")
abstract = st.text_area("Abstract", key="abstract_input")

def predict(text):
    inputs = tok(text, truncation=True, max_length=256, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.sigmoid(logits).numpy()

    # перенормировка и сортировка
    norm_probs = probs / probs.sum()
    order = norm_probs.argsort()[::-1]

    cumulative = 0
    results = []

    for i in order:
        label = id2label[int(i)]
        prob = float(probs[i])
        norm_prob = float(norm_probs[i])
        results.append((label, prob, norm_prob))
        cumulative += norm_prob
        if cumulative >= 0.95:
            break

    return results

if st.button("Классификация") and (title or abstract):
    text = (title.strip() + ". " + abstract.strip()).strip(". ")
    results = predict(text)

    st.subheader("Результат классификации")
    for label, prob, norm_prob in results:
        st.write(f"**{label}** — {prob:.1%}")
        st.progress(prob)

if st.button("Пример классификации"):
    st.session_state["title_input"] = "Attention Is All You Need"
    st.session_state["abstract_input"] = "The dominant sequence transduction models..."
    st.rerun()
