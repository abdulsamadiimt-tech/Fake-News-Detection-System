"""Text cleaning (NLTK) and simple writing-style features, shared by train.py and app.py."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ("stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

# "reuters" is removed on purpose: real articles in the dataset often start with
# "(Reuters)", which would let the model cheat instead of learning writing patterns.
STOP = set(stopwords.words("english")) | {"reuters"}
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    """Lowercase, drop links/punctuation/numbers/stopwords, then lemmatize."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(words)


def style_features(text):
    """Simple 'text pattern' numbers measured on the raw text."""
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    return {
        "word_count": len(text.split()),
        "exclamations": text.count("!"),
        "caps_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
    }
