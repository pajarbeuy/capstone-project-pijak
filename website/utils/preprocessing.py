import re
from functools import lru_cache

try:
    from nltk.corpus import stopwords
except Exception:
    stopwords = None

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
except Exception:
    StemmerFactory = None


KAMUS_NORMALISASI = {
    "bgt": "banget",
    "bget": "banget",
    "bngt": "banget",
    "gak": "tidak",
    "ga": "tidak",
    "gk": "tidak",
    "tdk": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "bs": "bisa",
    "bisa": "bisa",
    "ok": "oke",
    "oke": "oke",
    "oks": "oke",
    "sy": "saya",
    "aku": "saya",
    "gw": "saya",
    "gue": "saya",
    "tp": "tapi",
    "tapi": "tapi",
    "krn": "karena",
    "karna": "karena",
    "krena": "karena",
    "utk": "untuk",
    "u": "untuk",
    "dr": "dari",
    "yg": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "dlm": "dalam",
    "sdh": "sudah",
    "udah": "sudah",
    "udh": "sudah",
    "blm": "belum",
    "blum": "belum",
    "jd": "jadi",
    "jadi": "jadi",
    "sm": "sama",
    "jg": "juga",
    "cb": "coba",
    "beli": "beli",
    "pak": "bapak",
    "bu": "ibu",
    "mantap": "mantap",
    "mantabs": "mantap",
    "mantul": "mantap",
    "keren": "bagus",
    "top": "bagus",
    "jos": "bagus",
    "oke banget": "sangat oke",
    "ancur": "hancur",
    "anjir": "tidak menyenangkan",
    "nyesel": "menyesal",
    "zonk": "mengecewakan",
    "jelek": "buruk",
    "ongkir": "ongkos kirim",
    "packing": "kemasan",
    "packingnya": "kemasannya",
    "seller": "penjual",
    "cod": "bayar di tempat",
    "ori": "original",
    "asli": "original",
    "sesuai pict": "sesuai gambar",
}

STOPWORDS_TAMBAHAN = {
    "ya",
    "yah",
    "iya",
    "oiya",
    "oh",
    "ah",
    "eh",
    "wah",
    "nih",
    "sih",
    "deh",
    "dong",
    "kok",
    "aja",
    "saja",
    "lah",
    "dah",
    "udah",
    "emang",
    "memang",
    "hehe",
    "haha",
    "hehehe",
    "hm",
    "hmm",
    "yuk",
    "yg",
    "utk",
    "dr",
    "jg",
    "sy",
    "lg",
    "lagi",
    "nah",
    "mah",
}

KATA_PENTING = {
    "tidak",
    "belum",
    "kurang",
    "bukan",
    "jangan",
    "bagus",
    "buruk",
    "baik",
    "jelek",
    "puas",
    "kecewa",
    "rusak",
    "hancur",
    "mantap",
    "sesuai",
}

FALLBACK_STOPWORDS = {
    "ada",
    "adalah",
    "agar",
    "akan",
    "aku",
    "anda",
    "apa",
    "atau",
    "bagai",
    "bagi",
    "dalam",
    "dan",
    "dari",
    "dengan",
    "di",
    "dia",
    "ini",
    "itu",
    "jadi",
    "juga",
    "karena",
    "ke",
    "kita",
    "mereka",
    "oleh",
    "pada",
    "saya",
    "sebagai",
    "untuk",
    "yang",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F9FF"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    words = text.split()
    return " ".join(KAMUS_NORMALISASI.get(word, word) for word in words)


@lru_cache(maxsize=1)
def get_stopwords() -> set[str]:
    if stopwords is not None:
        try:
            words = set(stopwords.words("indonesian"))
        except LookupError:
            words = set(FALLBACK_STOPWORDS)
    else:
        words = set(FALLBACK_STOPWORDS)

    words.update(STOPWORDS_TAMBAHAN)
    return words - KATA_PENTING


def remove_stopwords(text: str) -> str:
    stop_words = get_stopwords()
    return " ".join(word for word in text.split() if word not in stop_words)


@lru_cache(maxsize=1)
def get_stemmer():
    if StemmerFactory is None:
        return None
    return StemmerFactory().create_stemmer()


def stem_text(text: str) -> str:
    stemmer = get_stemmer()
    if stemmer is None:
        return text
    return " ".join(stemmer.stem(word) for word in text.split())


def preprocess_text(text: str) -> str:
    cleaned = clean_text(text)
    normalized = normalize_text(cleaned)
    no_stopwords = remove_stopwords(normalized)
    return stem_text(no_stopwords)
