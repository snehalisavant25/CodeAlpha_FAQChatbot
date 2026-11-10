import streamlit as st

from chatbot import USE_NLTK, FAQBot

st.set_page_config(page_title="CodeAlpha FAQ Chatbot", page_icon="🤖")


@st.cache_resource
def load_bot():
    return FAQBot("faqs.json")


bot = load_bot()

st.title("🤖 CodeAlpha Internship FAQ Chatbot")
st.caption("Ask anything about the CodeAlpha AI internship: tasks, certificates, submission, contact.")

with st.sidebar:
    st.header("Try asking")
    samples = [
        "How many tasks must I complete?",
        "Will I get a certificate?",
        "How do I submit my work?",
        "What should I name my GitHub repo?",
        "How can I contact CodeAlpha?",
    ]
    clicked = None
    for s in samples:
        if st.button(s, use_container_width=True):
            clicked = s
    show_debug = st.toggle("Show matched question and score", value=True)
    st.caption(f"NLP: {'NLTK' if USE_NLTK else 'built-in fallback'} + TF-IDF + cosine similarity")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm the CodeAlpha FAQ bot. What would you like to know?"}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Type your question...") or clicked
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    answer, matched, score, suggestions = bot.get_answer(user_text)
    if answer:
        reply = answer
        if show_debug and matched:
            reply += f"\n\n*Matched FAQ: \"{matched}\" (similarity {score:.2f})*"
    else:
        reply = "Sorry, I couldn't find a good answer to that. Try rephrasing your question."
        if suggestions:
            reply += "\n\nDid you mean:\n" + "\n".join(f"- {q}" for q in suggestions)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)