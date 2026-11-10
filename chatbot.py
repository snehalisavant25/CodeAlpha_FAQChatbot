import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- NLP setup ----------
# RegexpTokenizer and PorterStemmer are part of NLTK and need NO data downloads,
# so they work even on restricted networks.
EXTRA_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "and",
    "or", "do", "does", "did", "i", "me", "my", "we", "you", "your", "it", "its", "this",
    "that", "can", "will", "would", "should", "what", "how", "where", "which", "who",
    "when", "there", "any", "have", "has", "with", "about", "at", "by", "from", "as",
    "tell", "please", "want", "know",
}

try:
    from nltk.stem import PorterStemmer
    from nltk.tokenize import RegexpTokenizer

    TOKENIZER = RegexpTokenizer(r"[a-z0-9]+")
    STEMMER = PorterStemmer()
    USE_NLTK = True
except Exception:
    USE_NLTK = False

STOP = set(EXTRA_STOP)
if USE_NLTK:
    try:  # use NLTK's full stopword list too, if its data happens to be installed
        from nltk.corpus import stopwords

        STOP |= set(stopwords.words("english"))
    except Exception:
        pass


def preprocess(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords, stem."""
    text = text.lower()
    if USE_NLTK:
        tokens = TOKENIZER.tokenize(text)
        return " ".join(STEMMER.stem(t) for t in tokens if t not in STOP)
    tokens = re.findall(r"[a-z0-9]+", text)
    return " ".join(t for t in tokens if t not in STOP)


SMALL_TALK = {
    ("hi", "hello", "hey", "hii", "good morning", "good evening"):
        "Hello! I'm the CodeAlpha internship FAQ bot. Ask me about tasks, certificates, submission and more.",
    ("thanks", "thank you", "thx"): "You're welcome! Ask me anything else about the internship.",
    ("bye", "goodbye", "see you"): "Goodbye! Best of luck with your internship.",
}


class FAQBot:
    def __init__(self, faq_path="faqs.json", threshold=0.2):
        with open(faq_path, encoding="utf-8") as f:
            self.faqs = json.load(f)
        self.threshold = threshold
        # Question counts double so it outweighs the answer text when matching
        docs = [preprocess(f"{x['question']} {x['question']} {x['answer']}") for x in self.faqs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(docs)

    def small_talk(self, text):
        t = re.sub(r"[^a-z\s]", "", text.lower()).strip()
        for keys, reply in SMALL_TALK.items():
            if t in keys:
                return reply
        return None

    def get_answer(self, user_text):
        """Return (answer, matched_question, score, suggestions)."""
        talk = self.small_talk(user_text)
        if talk:
            return talk, None, 1.0, []
        cleaned = preprocess(user_text)
        if not cleaned:
            return None, None, 0.0, []
        scores = cosine_similarity(self.vectorizer.transform([cleaned]), self.matrix)[0]
        ranked = scores.argsort()[::-1]
        best = ranked[0]
        if scores[best] >= self.threshold:
            return self.faqs[best]["answer"], self.faqs[best]["question"], float(scores[best]), []
        suggestions = [self.faqs[i]["question"] for i in ranked[:3] if scores[i] > 0]
        return None, None, float(scores[best]), suggestions