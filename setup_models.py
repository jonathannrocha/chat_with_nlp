import spacy
import spacy.cli
import nltk
from pathlib import Path


models = ["en_core_web_lg", "pt_core_news_lg"]
for model in models:
    try:
        spacy.load(model)
    except OSError:
        spacy.cli.download(model)

nltk_data_path = Path(__file__).parent / "nltk_data"
nltk.data.path.append(str(nltk_data_path))

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=str(nltk_data_path))


try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", download_dir=str(nltk_data_path))


