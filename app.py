from pathlib import Path
import io
import json
import re
import shutil
import unicodedata

import joblib
import numpy as np
import pandas as pd
import pymupdf
import pytesseract
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer


# ============================================================
# STREAMLIT / PATHS
# ============================================================

st.set_page_config(
    page_title="TriCheck-LK",
    page_icon="🔎",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"


# ============================================================
# CONFIGURATION
# ============================================================

MIN_TEXT_CHARS = 500
MAX_CHUNK_CHARS = 1000

LANGUAGE_OCR_CODES = {
    "English": "eng",
    "Sinhala": "sin+eng",
    "Tamil": "tam+eng",
}

LANGUAGE_CONFIGS = {
    "English": {
        "anchor": "Sinhala",
        "reference": "Tamil",
    },
    "Sinhala": {
        "anchor": "English",
        "reference": "Tamil",
    },
    "Tamil": {
        "anchor": "English",
        "reference": "Sinhala",
    },
}

SAME_DOC_STRONG_SIMILARITY = 0.70
SAME_DOC_MIN_MEDIAN_SIMILARITY = 0.64
SAME_DOC_MIN_BIDIRECTIONAL_COVERAGE = 0.50
SAME_DOC_MIN_MUTUAL_COVERAGE = 0.25
SAME_DOC_MIN_LENGTH_RATIO = 0.40

FALLBACK_MEDIAN_SIMILARITY = 0.78
FALLBACK_BIDIRECTIONAL_COVERAGE = 0.70
FALLBACK_MUTUAL_COVERAGE = 0.35
FALLBACK_LENGTH_RATIO = 0.50


# ============================================================
# TESSERACT
# ============================================================

def configure_tesseract():

    tesseract_path = shutil.which(
        "tesseract"
    )

    if tesseract_path:

        pytesseract.pytesseract.tesseract_cmd = (
            tesseract_path
        )

        return True

    windows_path = Path(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if windows_path.exists():

        pytesseract.pytesseract.tesseract_cmd = str(
            windows_path
        )

        return True

    return False


TESSERACT_AVAILABLE = (
    configure_tesseract()
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_resources():

    detectors = {

        "English":
            joblib.load(
                MODEL_DIR
                / "tricheck_english_detector.joblib"
            ),

        "Sinhala":
            joblib.load(
                MODEL_DIR
                / "tricheck_sinhala_detector.joblib"
            ),

        "Tamil":
            joblib.load(
                MODEL_DIR
                / "tricheck_tamil_detector.joblib"
            ),
    }

    with open(
        MODEL_DIR
        / "tricheck_multilingual_config.json",
        "r",
        encoding="utf-8"
    ) as file:

        model_config = (
            json.load(
                file
            )
        )

    embedding_model = (
        SentenceTransformer(
            model_config[
                "embedding_model"
            ]
        )
    )

    return (
        detectors,
        model_config,
        embedding_model
    )


try:

    (
        detectors,
        model_config,
        embedding_model
    ) = load_resources()

except Exception as error:

    st.error(
        "TriCheck-LK could not load the trained models. "
        f"Technical details: {error}"
    )

    st.stop()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = (
        unicodedata.normalize(
            "NFC",
            str(
                text
            )
        )
    )

    text = "".join(
        char
        for char in text
        if (
            unicodedata.category(
                char
            )[0]
            != "C"
            or char
            in "\n\t\u200c\u200d"
        )
    )

    text = (
        text.replace(
            "\t",
            " "
        )
    )

    text = re.sub(
        r"[ ]+",
        " ",
        text
    )

    text = re.sub(
        r"\n\s*\n+",
        "\n",
        text
    )

    return (
        text.strip()
    )


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_text_language(text):

    counts = {
        "English": 0,
        "Sinhala": 0,
        "Tamil": 0,
    }

    for char in str(
        text
    ):

        code = ord(
            char
        )

        if (
            "A"
            <= char
            <= "Z"
            or
            "a"
            <= char
            <= "z"
        ):

            counts[
                "English"
            ] += 1

        elif (
            0x0D80
            <= code
            <= 0x0DFF
        ):

            counts[
                "Sinhala"
            ] += 1

        elif (
            0x0B80
            <= code
            <= 0x0BFF
        ):

            counts[
                "Tamil"
            ] += 1

    total_letters = sum(
        counts.values()
    )

    if total_letters == 0:

        return "Unknown"

    detected_language = max(
        counts,
        key=counts.get
    )

    dominance = (
        counts[
            detected_language
        ]
        / total_letters
    )

    if dominance < 0.60:

        return "Unknown"

    return detected_language


def script_range(
    language
):

    if language == "Sinhala":

        return (
            0x0D80,
            0x0DFF
        )

    if language == "Tamil":

        return (
            0x0B80,
            0x0BFF
        )

    return None


# ============================================================
# TEXT QUALITY
# ============================================================

def text_quality_score(
    text,
    language
):
    """
    Score extracted text quality.

    A key corruption signal for Sinhala/Tamil PDFs is an
    isolated Unicode combining mark at the beginning of a token.
    PDF text layers with broken complex-script shaping often
    produce many such tokens even though no literal replacement
    character is present.
    """

    text = clean_text(text)

    if not text:
        return -1000.0

    score = min(len(text), 5000) / 5000.0

    # Obvious corruption characters.
    score -= text.count("�") * 5.0
    score -= text.count("◌") * 5.0
    score -= text.count("\x00") * 2.0

    tokens = re.findall(r"\S+", text)

    # A token starting with a combining mark indicates that the
    # base character and its dependent sign were separated by
    # the PDF text layer. This is common in damaged Tamil/Sinhala
    # extraction and is exactly what renders as dotted circles.
    orphan_mark_count = sum(
        1
        for token in tokens
        if token
        and unicodedata.category(token[0]).startswith("M")
    )

    orphan_mark_ratio = (
        orphan_mark_count / len(tokens)
        if tokens
        else 0.0
    )

    score -= orphan_mark_count * 0.12
    score -= orphan_mark_ratio * 8.0

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if not letters:
        return score - 5.0

    if language == "English":
        latin_letters = sum(
            1
            for char in letters
            if (
                "A" <= char <= "Z"
                or
                "a" <= char <= "z"
            )
        )

        score += 4.0 * (
            latin_letters / len(letters)
        )

        return score

    script = script_range(language)

    if script is None:
        return score

    start, end = script

    native_letters = sum(
        1
        for char in text
        if start <= ord(char) <= end
    )

    score += 4.0 * (
        native_letters / max(len(letters), 1)
    )

    # Penalize suspicious one-character native-script tokens.
    native_tokens = 0
    one_char_native = 0

    for token in tokens:
        native_chars = [
            char
            for char in token
            if start <= ord(char) <= end
        ]

        if native_chars:
            native_tokens += 1
            if len(token) == 1:
                one_char_native += 1

    if native_tokens:
        score -= 2.0 * (
            one_char_native / native_tokens
        )

    return score



def choose_best_extraction(
    direct_text,
    ocr_text,
    language
):

    direct_clean = (
        clean_text(
            direct_text
        )
    )

    ocr_clean = (
        clean_text(
            ocr_text
        )
    )

    if not ocr_clean:

        return (
            direct_clean,
            "DIRECT_TEXT"
        )

    if not direct_clean:

        return (
            ocr_clean,
            "OCR_TEXT"
        )

    direct_score = (
        text_quality_score(
            direct_clean,
            language
        )
    )

    ocr_score = (
        text_quality_score(
            ocr_clean,
            language
        )
    )

    if (
        ocr_score
        >
        direct_score
        + 0.20
    ):

        return (
            ocr_clean,
            "OCR_TEXT"
        )

    return (
        direct_clean,
        "DIRECT_TEXT"
    )


# ============================================================
# DIRECT PDF EXTRACTION
# ============================================================

def extract_direct_text(
    pdf_bytes
):

    document = None

    try:

        document = (
            pymupdf.open(
                stream=pdf_bytes,
                filetype="pdf"
            )
        )

        pages = []

        for page in document:

            page_text = (
                page.get_text(
                    "text"
                )
            )

            if page_text:

                pages.append(
                    page_text
                )

        extracted_text = (
            "\n".join(
                pages
            )
            .strip()
        )

        if (
            len(
                extracted_text
            )
            >= MIN_TEXT_CHARS
        ):

            return (
                extracted_text,
                "DIRECT_TEXT"
            )

        if extracted_text:

            return (
                extracted_text,
                "LOW_TEXT"
            )

        return (
            "",
            "EMPTY"
        )

    except Exception as error:

        print(
            "Direct extraction error:",
            error
        )

        return (
            "",
            "ERROR"
        )

    finally:

        if document is not None:

            document.close()


# ============================================================
# OCR
# ============================================================

def extract_ocr_text(
    pdf_bytes,
    language,
    max_pages=None,
    dpi=350
):

    if not TESSERACT_AVAILABLE:

        return ""

    document = None

    try:

        document = (
            pymupdf.open(
                stream=pdf_bytes,
                filetype="pdf"
            )
        )

        language_code = (
            LANGUAGE_OCR_CODES[
                language
            ]
        )

        page_count = len(
            document
        )

        if max_pages is not None:

            page_count = min(
                page_count,
                max_pages
            )

        pages = []

        for page_index in range(
            page_count
        ):

            page = (
                document[
                    page_index
                ]
            )

            pixmap = (
                page.get_pixmap(
                    dpi=dpi,
                    alpha=False
                )
            )

            image = (
                Image.open(
                    io.BytesIO(
                        pixmap.tobytes(
                            "png"
                        )
                    )
                )
            )

            page_text = (
                pytesseract
                .image_to_string(
                    image,
                    lang=language_code,
                    config=(
                        "--oem 3 "
                        "--psm 3 "
                        "-c preserve_interword_spaces=1"
                    )
                )
            )

            if page_text.strip():

                pages.append(
                    page_text.strip()
                )

        return (
            "\n".join(
                pages
            )
            .strip()
        )

    except Exception as error:

        print(
            f"OCR extraction error ({language}):",
            error
        )

        return ""

    finally:

        if document is not None:

            document.close()


def detect_language_with_ocr(
    pdf_bytes
):

    if not TESSERACT_AVAILABLE:

        return "Unknown"

    document = None

    try:

        document = (
            pymupdf.open(
                stream=pdf_bytes,
                filetype="pdf"
            )
        )

        detected_text = ""

        for page_index in range(
            min(
                len(
                    document
                ),
                2
            )
        ):

            page = (
                document[
                    page_index
                ]
            )

            pixmap = (
                page.get_pixmap(
                    dpi=200,
                    alpha=False
                )
            )

            image = (
                Image.open(
                    io.BytesIO(
                        pixmap.tobytes(
                            "png"
                        )
                    )
                )
            )

            page_text = (
                pytesseract
                .image_to_string(
                    image,
                    lang="eng+sin+tam",
                    config=(
                        "--oem 3 "
                        "--psm 3"
                    )
                )
            )

            detected_text += (
                page_text
                + "\n"
            )

        if not detected_text.strip():

            return "Unknown"

        return (
            detect_text_language(
                detected_text
            )
        )

    except Exception as error:

        print(
            "Language detection OCR error:",
            error
        )

        return "Unknown"

    finally:

        if document is not None:

            document.close()


# ============================================================
# HYBRID PDF EXTRACTION
# ============================================================
def needs_ocr_fallback(
    text,
    language
):
    """
    Return True when the embedded PDF text layer appears damaged.

    In Sinhala/Tamil, malformed PDF text frequently separates a
    dependent vowel/combining sign from its base character. The
    string itself may not contain the literal dotted-circle symbol;
    the UI draws the dotted circle when an isolated combining mark
    is rendered.
    """

    text = clean_text(text)

    if not text:
        return True

    if "�" in text or "◌" in text or "\x00" in text:
        return True

    tokens = re.findall(r"\S+", text)

    orphan_mark_count = sum(
        1
        for token in tokens
        if token
        and unicodedata.category(token[0]).startswith("M")
    )

    orphan_mark_ratio = (
        orphan_mark_count / len(tokens)
        if tokens
        else 0.0
    )

    # Require both an absolute count and a ratio so that one
    # unusual character does not unnecessarily trigger OCR.
    if (
        orphan_mark_count >= 3
        and
        orphan_mark_ratio >= 0.005
    ):
        return True

    if language == "Tamil":
        tamil_chars = sum(
            1
            for char in text
            if 0x0B80 <= ord(char) <= 0x0BFF
        )
        if tamil_chars < 50:
            return True

    if language == "Sinhala":
        sinhala_chars = sum(
            1
            for char in text
            if 0x0D80 <= ord(char) <= 0x0DFF
        )
        if sinhala_chars < 50:
            return True

    return False



def extract_pdf_text(
    uploaded_file,
    language
):

    pdf_bytes = (
        uploaded_file.getvalue()
    )

    raw_text, status = (
        extract_direct_text(
            pdf_bytes
        )
    )

    if status == "ERROR":

        return {
            "text":
                "",

            "method":
                "ERROR",

            "raw_characters":
                0,

            "characters":
                0,

            "detected_language":
                "Unknown",
        }

    direct_detected_language = (
        detect_text_language(
            raw_text
        )
        if raw_text.strip()
        else "Unknown"
    )

    detected_language = (
        direct_detected_language
    )

    if (
        detected_language
        == "Unknown"
        and
        TESSERACT_AVAILABLE
    ):

        detected_language = (
            detect_language_with_ocr(
                pdf_bytes
            )
        )

    if (
        detected_language
        != "Unknown"
        and
        detected_language
        != language
    ):

        cleaned = clean_text(
            raw_text
        )

        return {
            "text":
                cleaned,

            "method":
                "LANGUAGE_MISMATCH",

            "raw_characters":
                len(
                    raw_text
                ),

            "characters":
                len(
                    cleaned
                ),

            "detected_language":
                detected_language,
        }

    if detected_language == "Unknown":

        cleaned = clean_text(
            raw_text
        )

        return {
            "text":
                cleaned,

            "method":
                "UNVERIFIED_LANGUAGE",

            "raw_characters":
                len(
                    raw_text
                ),

            "characters":
                len(
                    cleaned
                ),

            "detected_language":
                "Unknown",
        }

    if (
        language
        == "English"
        and
        status
        == "DIRECT_TEXT"
    ):

        cleaned = clean_text(
            raw_text
        )

        return {
            "text":
                cleaned,

            "method":
                "DIRECT_TEXT",

            "raw_characters":
                len(
                    raw_text
                ),

            "characters":
                len(
                    cleaned
                ),

            "detected_language":
                detected_language,
        }

   

       
        

       

        
        # ========================================================
    # DIRECT TEXT FIRST
    # OCR ONLY IF THE DIRECT TEXT LOOKS DAMAGED
    # ========================================================

    if status == "DIRECT_TEXT":

        cleaned_direct_text = (
            clean_text(
                raw_text
            )
        )

        if (
            language in {
                "Sinhala",
                "Tamil"
            }
            and
            TESSERACT_AVAILABLE
            and
            needs_ocr_fallback(
                cleaned_direct_text,
                language
            )
        ):

            raw_ocr_text = (
                extract_ocr_text(
                    pdf_bytes,
                    language
                )
            )

            cleaned_ocr_text = (
                clean_text(
                    raw_ocr_text
                )
            )

            direct_score = (
                text_quality_score(
                    cleaned_direct_text,
                    language
                )
            )

            ocr_score = (
                text_quality_score(
                    cleaned_ocr_text,
                    language
                )
            )

            if (
                cleaned_ocr_text
                and
                ocr_score
                > direct_score
            ):

                return {
                    "text":
                        cleaned_ocr_text,

                    "method":
                        "OCR_TEXT",

                    "raw_characters":
                        len(
                            raw_ocr_text
                        ),

                    "characters":
                        len(
                            cleaned_ocr_text
                        ),

                    "detected_language":
                        language,
                }

        return {
            "text":
                cleaned_direct_text,

            "method":
                "DIRECT_TEXT",

            "raw_characters":
                len(
                    raw_text
                ),

            "characters":
                len(
                    cleaned_direct_text
                ),

            "detected_language":
                detected_language,
        }
    
    if status in {
        "LOW_TEXT",
        "EMPTY"
    }:

        raw_ocr_text = (
            extract_ocr_text(
                pdf_bytes,
                language
            )
            if TESSERACT_AVAILABLE
            else ""
        )

        cleaned_ocr_text = (
            clean_text(
                raw_ocr_text
            )
        )

        if (
            len(
                cleaned_ocr_text
            )
            >= MIN_TEXT_CHARS
        ):

            ocr_language = (
                detect_text_language(
                    cleaned_ocr_text
                )
            )

            if (
                ocr_language
                != "Unknown"
                and
                ocr_language
                != language
            ):

                return {
                    "text":
                        cleaned_ocr_text,

                    "method":
                        "LANGUAGE_MISMATCH",

                    "raw_characters":
                        len(
                            raw_ocr_text
                        ),

                    "characters":
                        len(
                            cleaned_ocr_text
                        ),

                    "detected_language":
                        ocr_language,
                }

            return {
                "text":
                    cleaned_ocr_text,

                "method":
                    "OCR_TEXT",

                "raw_characters":
                    len(
                        raw_ocr_text
                    ),

                "characters":
                    len(
                        cleaned_ocr_text
                    ),

                "detected_language":
                    (
                        language
                        if ocr_language
                        == "Unknown"
                        else ocr_language
                    ),
            }

        cleaned_direct_text = (
            clean_text(
                raw_text
            )
        )

        if cleaned_direct_text:

            return {
                "text":
                    cleaned_direct_text,

                "method":
                    "DIRECT_TEXT",

                "raw_characters":
                    len(
                        raw_text
                    ),

                "characters":
                    len(
                        cleaned_direct_text
                    ),

                "detected_language":
                    detected_language,
            }

        return {
            "text":
                cleaned_ocr_text,

            "method":
                "UNUSABLE",

            "raw_characters":
                len(
                    raw_ocr_text
                ),

            "characters":
                len(
                    cleaned_ocr_text
                ),

            "detected_language":
                detected_language,
        }

    return {
        "text":
            "",

        "method":
            "ERROR",

        "raw_characters":
            0,

        "characters":
            0,

        "detected_language":
            "Unknown",
    }


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text,
    max_chars=MAX_CHUNK_CHARS
):

    text = (
        re.sub(
            r"\s+",
            " ",
            text
        )
        .strip()
    )

    raw_sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = []

    index = 0

    while (
        index
        < len(
            raw_sentences
        )
    ):

        current = (
            raw_sentences[
                index
            ]
            .strip()
        )

        if (
            re.fullmatch(
                r"\d+(?:\.\d+)*\.",
                current
            )
            and
            index + 1
            < len(
                raw_sentences
            )
        ):

            sentences.append(
                current
                + " "
                + raw_sentences[
                    index + 1
                ].strip()
            )

            index += 2

        else:

            if current:

                sentences.append(
                    current
                )

            index += 1

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        sentence = (
            sentence.strip()
        )

        if not sentence:

            continue

        if (
            len(
                sentence
            )
            > max_chars
        ):

            for word in sentence.split():

                proposed = (
                    current_chunk
                    + " "
                    + word
                ).strip()

                if (
                    len(
                        proposed
                    )
                    <= max_chars
                ):

                    current_chunk = (
                        proposed
                    )

                else:

                    if current_chunk:

                        chunks.append(
                            current_chunk.strip()
                        )

                    current_chunk = (
                        word
                    )

            continue

        proposed = (
            current_chunk
            + " "
            + sentence
        ).strip()

        if (
            len(
                proposed
            )
            <= max_chars
        ):

            current_chunk = (
                proposed
            )

        else:

            if current_chunk:

                chunks.append(
                    current_chunk.strip()
                )

            current_chunk = (
                sentence
            )

    if current_chunk:

        chunks.append(
            current_chunk.strip()
        )

    return chunks


# ============================================================
# DOCUMENT VALIDATION
# ============================================================

def validate_document(
    result,
    chunks,
    language
):

    if result[
        "method"
    ] == "ERROR":

        return (
            False,
            f"{language} PDF could not be processed."
        )

    detected_language = (
        result.get(
            "detected_language",
            "Unknown"
        )
    )

    if (
        result[
            "method"
        ]
        == "LANGUAGE_MISMATCH"
        or
        (
            detected_language
            != "Unknown"
            and
            detected_language
            != language
        )
    ):

        return (
            False,
            f"The file uploaded in the {language} section "
            f"appears to be a {detected_language} document. "
            f"Please upload the correct {language} version."
        )

    if (
        result[
            "method"
        ]
        == "UNUSABLE"
    ):

        return (
            False,
            f"{language} PDF does not contain enough usable text."
        )

    if (
        detected_language
        == "Unknown"
    ):

        return (
            False,
            f"The language of the file uploaded in the "
            f"{language} section could not be verified. "
            f"Please upload a clear {language} version."
        )

    if not result[
        "text"
    ].strip():

        return (
            False,
            f"No usable text was extracted from the "
            f"{language} PDF."
        )

    if not chunks:

        return (
            False,
            f"No text chunks could be created from the "
            f"{language} PDF."
        )

    return (
        True,
        ""
    )


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(
    chunks,
    model
):

    model_inputs = [
        "query: "
        + chunk
        for chunk in chunks
    ]

    return (
        model.encode(
            model_inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    )


# ============================================================
# DTW
# ============================================================

def dtw_align(
    source_embeddings,
    target_embeddings
):

    similarity_matrix = (
        np.matmul(
            source_embeddings,
            target_embeddings.T
        )
    )

    cost_matrix = (
        1.0
        - similarity_matrix
    )

    (
        source_count,
        target_count
    ) = (
        cost_matrix.shape
    )

    dtw = np.full(
        (
            source_count + 1,
            target_count + 1
        ),
        np.inf
    )

    dtw[
        0,
        0
    ] = 0.0

    for i in range(
        1,
        source_count + 1
    ):

        for j in range(
            1,
            target_count + 1
        ):

            dtw[
                i,
                j
            ] = (
                cost_matrix[
                    i - 1,
                    j - 1
                ]
                +
                min(
                    dtw[
                        i - 1,
                        j
                    ],

                    dtw[
                        i,
                        j - 1
                    ],

                    dtw[
                        i - 1,
                        j - 1
                    ]
                )
            )

    i = source_count
    j = target_count

    path = []

    while (
        i > 0
        and
        j > 0
    ):

        path.append(
            (
                i - 1,
                j - 1,
                float(
                    similarity_matrix[
                        i - 1,
                        j - 1
                    ]
                )
            )
        )

        previous_costs = [
            dtw[
                i - 1,
                j - 1
            ],

            dtw[
                i - 1,
                j
            ],

            dtw[
                i,
                j - 1
            ],
        ]

        move = int(
            np.argmin(
                previous_costs
            )
        )

        if move == 0:

            i -= 1
            j -= 1

        elif move == 1:

            i -= 1

        else:

            j -= 1

    while i > 0:

        path.append(
            (
                i - 1,
                0,
                float(
                    similarity_matrix[
                        i - 1,
                        0
                    ]
                )
            )
        )

        i -= 1

    while j > 0:

        path.append(
            (
                0,
                j - 1,
                float(
                    similarity_matrix[
                        0,
                        j - 1
                    ]
                )
            )
        )

        j -= 1

    path.reverse()

    return (
        path,
        similarity_matrix
    )


# ============================================================
# DOCUMENT PAIR METRICS
# ============================================================

def document_pair_metrics(
    embeddings_a,
    embeddings_b
):

    if (
        embeddings_a is None
        or
        embeddings_b is None
        or
        len(
            embeddings_a
        )
        == 0
        or
        len(
            embeddings_b
        )
        == 0
    ):

        return {
            "median_similarity":
                0.0,

            "bidirectional_coverage":
                0.0,

            "mutual_coverage":
                0.0,

            "length_ratio":
                0.0,
        }

    similarity_matrix = (
        np.matmul(
            embeddings_a,
            embeddings_b.T
        )
    )

    best_b_for_a = (
        np.argmax(
            similarity_matrix,
            axis=1
        )
    )

    best_scores_a = (
        np.max(
            similarity_matrix,
            axis=1
        )
    )

    best_a_for_b = (
        np.argmax(
            similarity_matrix,
            axis=0
        )
    )

    best_scores_b = (
        np.max(
            similarity_matrix,
            axis=0
        )
    )

    combined = (
        np.concatenate(
            [
                best_scores_a,
                best_scores_b
            ]
        )
    )

    median_similarity = (
        float(
            np.median(
                combined
            )
        )
    )

    coverage_a = (
        float(
            np.mean(
                best_scores_a
                >= SAME_DOC_STRONG_SIMILARITY
            )
        )
    )

    coverage_b = (
        float(
            np.mean(
                best_scores_b
                >= SAME_DOC_STRONG_SIMILARITY
            )
        )
    )

    bidirectional_coverage = (
        min(
            coverage_a,
            coverage_b
        )
    )

    mutual_matches = 0

    for (
        index_a,
        index_b
    ) in enumerate(
        best_b_for_a
    ):

        if (
            best_a_for_b[
                index_b
            ]
            == index_a
            and
            similarity_matrix[
                index_a,
                index_b
            ]
            >= SAME_DOC_STRONG_SIMILARITY
        ):

            mutual_matches += 1

    minimum_chunk_count = (
        min(
            len(
                embeddings_a
            ),
            len(
                embeddings_b
            )
        )
    )

    maximum_chunk_count = (
        max(
            len(
                embeddings_a
            ),
            len(
                embeddings_b
            )
        )
    )

    mutual_coverage = (
        mutual_matches
        / minimum_chunk_count
        if minimum_chunk_count
        else 0.0
    )

    length_ratio = (
        minimum_chunk_count
        / maximum_chunk_count
        if maximum_chunk_count
        else 0.0
    )

    return {
        "median_similarity":
            median_similarity,

        "bidirectional_coverage":
            bidirectional_coverage,

        "mutual_coverage":
            mutual_coverage,

        "length_ratio":
            length_ratio,
    }


# ============================================================
# DOCUMENT REFERENCE EXTRACTION
# ============================================================

def normalize_reference(
    reference
):

    if reference is None:

        return None

    reference = (
        str(
            reference
        )
        .upper()
    )

    reference = (
        reference
        .replace(
            "–",
            "-"
        )
        .replace(
            "—",
            "-"
        )
        .replace(
            "−",
            "-"
        )
    )

    reference = re.sub(
        r"\s+",
        "",
        reference
    )

    return (
        reference.strip(
            ".,;:()[]{}"
        )
    )


def extract_document_references(
    chunks
):

    if not chunks:

        return []

    text = (
        clean_text(
            " ".join(
                chunks[
                    :8
                ]
            )
        )
    )

    text = (
        text
        .replace(
            "–",
            "-"
        )
        .replace(
            "—",
            "-"
        )
        .replace(
            "−",
            "-"
        )
    )

    text = re.sub(
        r"\s*([/\-_])\s*",
        r"\1",
        text
    )

    label_patterns = [

        r"circular\s*"
        r"(?:no\.?|number|ref(?:erence)?(?:\s*no\.?)?)",

        r"(?:reference|ref)\s*"
        r"(?:no\.?|number)",

        r"file\s*"
        r"(?:no\.?|number)",

        r"චක්‍රලේඛ(?:ය)?\s*"
        r"(?:අංකය|අංක)",

        r"යොමු\s*"
        r"(?:අංකය|අංක)",

        r"சுற்றறிக்கை\s*"
        r"(?:இலக்கம்|எண்|இல\.?)",

        r"(?:குறிப்பு|கோப்பு)\s*"
        r"(?:இலக்கம்|எண்)",
    ]

    reference_pattern = (
        r"([A-Za-z0-9]"
        r"[A-Za-z0-9./_()\-]{2,45})"
    )

    references = []

    for label_pattern in (
        label_patterns
    ):

        pattern = (
            rf"(?:{label_pattern})"
            rf"\s*[:\-]?\s*"
            rf"{reference_pattern}"
        )

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            reference = (
                normalize_reference(
                    match.group(
                        1
                    )
                )
            )

            if not reference:

                continue

            if not re.search(
                r"\d",
                reference
            ):

                continue

            upper_reference = (
                reference.upper()
            )

            if any(
                token
                in upper_reference
                for token in (
                    "HTTP",
                    "WWW.",
                    ".COM",
                    ".ORG",
                    ".LK",
                    ".PDF",
                )
            ):

                continue

            if (
                reference
                not in references
            ):

                references.append(
                    reference
                )

    return references


# ============================================================
# FILENAME SUPPORT
# ============================================================

def filename_document_key(
    filename
):

    if not filename:

        return None

    stem = (
        Path(
            filename
        )
        .stem
        .lower()
    )

    long_numbers = (
        re.findall(
            r"\d{4,}",
            stem
        )
    )

    if not long_numbers:

        return None

    return (
        "|".join(
            long_numbers
        )
    )


# ============================================================
# PAIR DECISIONS
# ============================================================

def pair_semantic_match(
    metrics
):

    return (
        metrics[
            "median_similarity"
        ]
        >= SAME_DOC_MIN_MEDIAN_SIMILARITY
        and
        metrics[
            "bidirectional_coverage"
        ]
        >= SAME_DOC_MIN_BIDIRECTIONAL_COVERAGE
        and
        metrics[
            "mutual_coverage"
        ]
        >= SAME_DOC_MIN_MUTUAL_COVERAGE
        and
        metrics[
            "length_ratio"
        ]
        >= SAME_DOC_MIN_LENGTH_RATIO
    )


def pair_strong_semantic_match(
    metrics
):

    return (
        metrics[
            "median_similarity"
        ]
        >= FALLBACK_MEDIAN_SIMILARITY
        and
        metrics[
            "bidirectional_coverage"
        ]
        >= FALLBACK_BIDIRECTIONAL_COVERAGE
        and
        metrics[
            "mutual_coverage"
        ]
        >= FALLBACK_MUTUAL_COVERAGE
        and
        metrics[
            "length_ratio"
        ]
        >= FALLBACK_LENGTH_RATIO
    )


def check_document_pair(
    chunks_a,
    embeddings_a,
    filename_a,
    chunks_b,
    embeddings_b,
    filename_b
):

    references_a = (
        extract_document_references(
            chunks_a
        )
    )

    references_b = (
        extract_document_references(
            chunks_b
        )
    )

    set_a = set(
        references_a
    )

    set_b = set(
        references_b
    )

    common_references = (
        set_a.intersection(
            set_b
        )
    )

    filename_key_a = (
        filename_document_key(
            filename_a
        )
    )

    filename_key_b = (
        filename_document_key(
            filename_b
        )
    )

    filename_match = (
        filename_key_a
        is not None
        and
        filename_key_b
        is not None
        and
        filename_key_a
        == filename_key_b
    )

    metrics = (
        document_pair_metrics(
            embeddings_a,
            embeddings_b
        )
    )

    semantic_match = (
        pair_semantic_match(
            metrics
        )
    )

    strong_semantic_match = (
        pair_strong_semantic_match(
            metrics
        )
    )

    references_available = (
        bool(
            set_a
        )
        and
        bool(
            set_b
        )
    )

    if references_available:

        reference_match = (
            bool(
                common_references
            )
        )

    else:

        reference_match = None

    if references_available:

        match = (
            bool(
                reference_match
                and
                semantic_match
            )
        )

        identity_confirmed = (
            match
        )

    elif filename_match:

        match = (
            bool(
                semantic_match
            )
        )

        identity_confirmed = (
            match
        )

    else:

        match = (
            bool(
                strong_semantic_match
            )
        )

        identity_confirmed = (
            False
        )

    return {
        "match":
            match,

        "identity_confirmed":
            identity_confirmed,

        "references_a":
            references_a,

        "references_b":
            references_b,

        "common_references":
            sorted(
                common_references
            ),

        "reference_match":
            reference_match,

        "filename_key_a":
            filename_key_a,

        "filename_key_b":
            filename_key_b,

        "filename_match":
            filename_match,

        "semantic_match":
            semantic_match,

        "strong_semantic_match":
            strong_semantic_match,

        "metrics":
            metrics,
    }


# ============================================================
# VERIFY ALL 3 PAIRS
# ============================================================

def verify_same_document(
    prepared
):

    en_si = (
        check_document_pair(
            prepared[
                "English"
            ][
                "chunks"
            ],
            prepared[
                "English"
            ][
                "embeddings"
            ],
            prepared[
                "English"
            ].get(
                "filename"
            ),
            prepared[
                "Sinhala"
            ][
                "chunks"
            ],
            prepared[
                "Sinhala"
            ][
                "embeddings"
            ],
            prepared[
                "Sinhala"
            ].get(
                "filename"
            ),
        )
    )

    en_ta = (
        check_document_pair(
            prepared[
                "English"
            ][
                "chunks"
            ],
            prepared[
                "English"
            ][
                "embeddings"
            ],
            prepared[
                "English"
            ].get(
                "filename"
            ),
            prepared[
                "Tamil"
            ][
                "chunks"
            ],
            prepared[
                "Tamil"
            ][
                "embeddings"
            ],
            prepared[
                "Tamil"
            ].get(
                "filename"
            ),
        )
    )

    si_ta = (
        check_document_pair(
            prepared[
                "Sinhala"
            ][
                "chunks"
            ],
            prepared[
                "Sinhala"
            ][
                "embeddings"
            ],
            prepared[
                "Sinhala"
            ].get(
                "filename"
            ),
            prepared[
                "Tamil"
            ][
                "chunks"
            ],
            prepared[
                "Tamil"
            ][
                "embeddings"
            ],
            prepared[
                "Tamil"
            ].get(
                "filename"
            ),
        )
    )

    pair_results = {

        "English-Sinhala":
            en_si,

        "English-Tamil":
            en_ta,

        "Sinhala-Tamil":
            si_ta,
    }

    same_document = all(
        result[
            "match"
        ]
        for result
        in pair_results.values()
    )

    print(
        "\n========== DOCUMENT PAIR CHECK =========="
    )

    for (
        pair,
        result
    ) in pair_results.items():

        metrics = (
            result[
                "metrics"
            ]
        )

        print(
            f"\n{pair}"
        )

        print(
            "References A:",
            result[
                "references_a"
            ]
        )

        print(
            "References B:",
            result[
                "references_b"
            ]
        )

        print(
            "Common references:",
            result[
                "common_references"
            ]
        )

        print(
            "Reference match:",
            result[
                "reference_match"
            ]
        )

        print(
            "Filename key A:",
            result[
                "filename_key_a"
            ]
        )

        print(
            "Filename key B:",
            result[
                "filename_key_b"
            ]
        )

        print(
            "Filename match:",
            result[
                "filename_match"
            ]
        )

        print(
            "Semantic match:",
            result[
                "semantic_match"
            ]
        )

        print(
            "Strong semantic match:",
            result[
                "strong_semantic_match"
            ]
        )

        print(
            "Identity confirmed:",
            result[
                "identity_confirmed"
            ]
        )

        print(
            "Median similarity:",
            round(
                metrics[
                    "median_similarity"
                ],
                3
            )
        )

        print(
            "Bidirectional coverage:",
            round(
                metrics[
                    "bidirectional_coverage"
                ],
                3
            )
        )

        print(
            "Mutual coverage:",
            round(
                metrics[
                    "mutual_coverage"
                ],
                3
            )
        )

        print(
            "Length ratio:",
            round(
                metrics[
                    "length_ratio"
                ],
                3
            )
        )

        print(
            "FINAL MATCH:",
            result[
                "match"
            ]
        )

    print(
        "\nALL DOCUMENTS CORRESPOND:",
        same_document
    )

    print(
        "=========================================\n"
    )

    return (
        same_document,
        {
            "pair_results":
                pair_results
        }
    )


# ============================================================
# IDENTIFY WHICH SINGLE LANGUAGE IS DIFFERENT
# ============================================================

def identify_mismatched_language(
    pair_results
):

    en_si = (
        pair_results[
            "English-Sinhala"
        ]
    )

    en_ta = (
        pair_results[
            "English-Tamil"
        ]
    )

    si_ta = (
        pair_results[
            "Sinhala-Tamil"
        ]
    )

    en_si_match = (
        en_si[
            "match"
        ]
    )

    en_ta_match = (
        en_ta[
            "match"
        ]
    )

    si_ta_match = (
        si_ta[
            "match"
        ]
    )

    if (
        en_si_match
        and
        en_ta_match
        and
        si_ta_match
    ):

        return None

    if (
        en_si[
            "identity_confirmed"
        ]
        and
        not en_ta_match
        and
        not si_ta_match
    ):

        return "Tamil"

    if (
        en_ta[
            "identity_confirmed"
        ]
        and
        not en_si_match
        and
        not si_ta_match
    ):

        return "Sinhala"

    if (
        si_ta[
            "identity_confirmed"
        ]
        and
        not en_si_match
        and
        not en_ta_match
    ):

        return "English"

    return "MULTIPLE"


# ============================================================
# SOURCE COVERAGE
# ============================================================

def get_source_coverage(
    path,
    similarity_matrix,
    source_count
):

    coverage = {
        i: []
        for i in range(
            source_count
        )
    }

    for (
        source_pos,
        target_pos,
        _
    ) in path:

        score = float(
            similarity_matrix[
                source_pos,
                target_pos
            ]
        )

        coverage[
            source_pos
        ].append(
            score
        )

    return [
        max(
            coverage[
                i
            ]
        )
        if coverage[
            i
        ]
        else np.nan
        for i in range(
            source_count
        )
    ]


def create_path_dataframe(
    path
):

    return pd.DataFrame(
        [
            {
                "source_position":
                    source_pos,

                "target_position":
                    target_pos,

                "source_chunk":
                    source_pos,

                "target_chunk":
                    target_pos,

                "similarity":
                    float(
                        similarity
                    ),
            }
            for (
                source_pos,
                target_pos,
                similarity
            ) in path
        ]
    )


def add_compression_feature(
    path_df
):

    target_counts = (
        path_df[
            "target_position"
        ]
        .value_counts()
    )

    result = (
        path_df.copy()
    )

    result[
        "compression_count"
    ] = (
        result[
            "target_position"
        ]
        .map(
            target_counts
        )
    )

    return result


# ============================================================
# FEATURE GENERATION
# ============================================================

def build_document_features(
    prepared,
    target_language
):

    analysis_config = (
        LANGUAGE_CONFIGS[
            target_language
        ]
    )

    anchor_language = (
        analysis_config[
            "anchor"
        ]
    )

    reference_language = (
        analysis_config[
            "reference"
        ]
    )

    anchor_chunks = (
        prepared[
            anchor_language
        ][
            "chunks"
        ]
    )

    target_chunks = (
        prepared[
            target_language
        ][
            "chunks"
        ]
    )

    anchor_embeddings = (
        prepared[
            anchor_language
        ][
            "embeddings"
        ]
    )

    target_embeddings = (
        prepared[
            target_language
        ][
            "embeddings"
        ]
    )

    reference_embeddings = (
        prepared[
            reference_language
        ][
            "embeddings"
        ]
    )

    (
        target_path,
        _
    ) = dtw_align(
        anchor_embeddings,
        target_embeddings
    )

    (
        reference_path,
        reference_similarity_matrix
    ) = dtw_align(
        anchor_embeddings,
        reference_embeddings
    )

    reference_scores = (
        get_source_coverage(
            reference_path,
            reference_similarity_matrix,
            len(
                anchor_chunks
            )
        )
    )

    path_df = (
        add_compression_feature(
            create_path_dataframe(
                target_path
            )
        )
    )

    source_max = max(
        len(
            anchor_chunks
        )
        - 1,
        1
    )

    target_max = max(
        len(
            target_chunks
        )
        - 1,
        1
    )

    path_df[
        "position_difference"
    ] = abs(
        path_df[
            "source_position"
        ]
        / source_max
        -
        path_df[
            "target_position"
        ]
        / target_max
    )

    target_features = (
        path_df
        .groupby(
            "source_chunk"
        )
        .agg(
            target_similarity=(
                "similarity",
                "max"
            ),

            compression_count=(
                "compression_count",
                "max"
            ),

            position_difference=(
                "position_difference",
                "max"
            ),
        )
        .reset_index()
    )

    reference_features = (
        pd.DataFrame(
            {
                "source_chunk":
                    list(
                        range(
                            len(
                                anchor_chunks
                            )
                        )
                    ),

                "reference_similarity":
                    reference_scores,
            }
        )
    )

    feature_df = (
        target_features
        .merge(
            reference_features,
            on="source_chunk",
            how="left"
        )
        .sort_values(
            "source_chunk"
        )
        .reset_index(
            drop=True
        )
    )

    feature_df[
        "compression_local_max"
    ] = (
        feature_df[
            "compression_count"
        ]
        .rolling(
            window=3,
            center=True,
            min_periods=1
        )
        .max()
    )

    feature_df[
        "cross_language_gap"
    ] = (
        feature_df[
            "reference_similarity"
        ]
        -
        feature_df[
            "target_similarity"
        ]
    )

    feature_df[
        "local_median"
    ] = (
        feature_df[
            "target_similarity"
        ]
        .rolling(
            window=5,
            center=True,
            min_periods=1
        )
        .median()
    )

    feature_df[
        "local_drop"
    ] = (
        feature_df[
            "local_median"
        ]
        -
        feature_df[
            "target_similarity"
        ]
    )

    best_target_mapping = (
        path_df
        .sort_values(
            "similarity",
            ascending=False
        )
        .drop_duplicates(
            "source_chunk"
        )[
            [
                "source_chunk",
                "target_chunk"
            ]
        ]
    )

    feature_df = (
        feature_df
        .merge(
            best_target_mapping,
            on="source_chunk",
            how="left"
        )
    )

    feature_df[
        "target_language"
    ] = target_language

    feature_df[
        "anchor_language"
    ] = anchor_language

    feature_df[
        "reference_language"
    ] = reference_language

    return feature_df


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_missing_information(
    feature_df,
    language
):
    """
    Run the trained detector and create the review decision.

    The model configuration threshold is retained, but the UI uses
    a conservative minimum review probability of 0.60. This keeps
    low-confidence model-only outputs from being presented as final
    review regions. High-confidence structural rules can still flag
    a region independently later.
    """

    language_model_config = (
        model_config["languages"][language]
    )

    feature_columns = (
        language_model_config["features"]
    )

    configured_threshold = float(
        language_model_config["threshold"]
    )

    # Conservative UI review threshold. The trained probability is
    # still preserved in missing_probability for inspection.
    effective_threshold = max(
        configured_threshold,
        0.60
    )

    X = feature_df[feature_columns].copy()

    if X.isna().any().any():
        raise ValueError(
            f"{language} features contain missing values."
        )

    detector = detectors[language]

    probabilities = detector.predict_proba(X)[:, 1]

    model_predictions = (
        probabilities >= effective_threshold
    )

    result_df = feature_df.copy()

    result_df["missing_probability"] = probabilities
    result_df["review_score"] = probabilities
    result_df["rule_flag"] = False
    result_df["potential_missing"] = model_predictions

    return (
        result_df,
        effective_threshold
    )

# ============================================================
# NUMBERED SECTION MISSING-INFORMATION CHECK
# ============================================================

def extract_numbered_sections(
    chunks
):
    """
    Detect numbered document sections robustly.

    Supports OCR variations such as:

    1. Section
    1 . Section
    1) Section
    1 ) Section
    1 - Section
    1: Section

    Avoids matching times such as:
    9.00
    4.30
    """

    section_map = {}

    pattern = re.compile(
        r"(?<!\d)"
        r"([1-9]\d?)"
        r"\s*"
        r"[\.\)\-:]"
        r"\s+"
        r"(?=[^\d])"
    )

    for (
        chunk_index,
        chunk
    ) in enumerate(
        chunks
    ):

        normalized_chunk = (
            str(
                chunk
            )
            .replace(
                "．",
                "."
            )
            .replace(
                "。 ",
                ". "
            )
            .replace(
                "–",
                "-"
            )
            .replace(
                "—",
                "-"
            )
        )

        for match in pattern.finditer(
            normalized_chunk
        ):

            section_number = int(
                match.group(
                    1
                )
            )

            # Only realistic circular sections
            if not (
                1
                <= section_number
                <= 30
            ):
                continue

            if (
                section_number
                not in section_map
            ):

                section_map[
                    section_number
                ] = chunk_index

    return section_map

def apply_numbered_section_rule(
    analysis_df,
    prepared,
    target_language
):
    """
    Detect a numbered section missing from one language.

    A section is considered reliable evidence when it exists
    in BOTH of the other language versions but is absent from
    the target language.

    Example:

    English:
        1 2 3 4 5 6 7

    Sinhala:
        1 2 3 4 5 6 7

    Tamil:
        1 2 3   5 6 7

    Result:
        Tamil Section 4 -> Review required
    """

    result_df = (
        analysis_df.copy()
    )

    config = (
        LANGUAGE_CONFIGS[
            target_language
        ]
    )

    anchor_language = (
        config[
            "anchor"
        ]
    )

    reference_language = (
        config[
            "reference"
        ]
    )


    # --------------------------------------------------------
    # EXTRACT SECTION NUMBERS
    # --------------------------------------------------------

    anchor_sections = (
        extract_numbered_sections(
            prepared[
                anchor_language
            ][
                "chunks"
            ]
        )
    )

    reference_sections = (
        extract_numbered_sections(
            prepared[
                reference_language
            ][
                "chunks"
            ]
        )
    )

    target_sections = (
        extract_numbered_sections(
            prepared[
                target_language
            ][
                "chunks"
            ]
        )
    )


    # --------------------------------------------------------
    # A section must appear in BOTH other languages.
    # --------------------------------------------------------

    confirmed_sections = (
        set(
            anchor_sections.keys()
        )
        &
        set(
            reference_sections.keys()
        )
    )
    candidate_missing_sections = sorted(
    confirmed_sections
    -
    set(
        target_sections.keys()
    )
    )

    missing_sections = []

    for section_number in candidate_missing_sections:

    # --------------------------------------------------------
    # A missing middle section is much stronger evidence.
    #
    # Example:
    # Target has 3 and 5 but not 4
    # -> strong evidence that section 4 is missing.
    # --------------------------------------------------------

        previous_exists = (
            section_number - 1
            in target_sections
        )

        next_exists = (
            section_number + 1
            in target_sections
        )

        if (
            previous_exists
            and
            next_exists
        ):

            missing_sections.append(
                section_number
            )

        


    # --------------------------------------------------------
    # MAKE REQUIRED OUTPUT COLUMNS
    # --------------------------------------------------------

    if (
        "review_score"
        not in result_df.columns
    ):

        result_df[
            "review_score"
        ] = (
            result_df[
                "missing_probability"
            ]
        )


    if (
        "rule_flag"
        not in result_df.columns
    ):

        result_df[
            "rule_flag"
        ] = False


    result_df[
        "structural_missing_section"
    ] = False


    result_df[
        "missing_section_numbers"
    ] = ""


    # --------------------------------------------------------
    # FLAG SOURCE REGIONS CONTAINING THE MISSING SECTION
    # --------------------------------------------------------

    for section_number in (
        missing_sections
    ):

        source_chunk = (
            anchor_sections[
                section_number
            ]
        )

        matching_rows = (
            result_df.index[
                result_df[
                    "source_chunk"
                ]
                == source_chunk
            ]
        )


        if len(
            matching_rows
        ) == 0:

            continue


        row_index = (
            matching_rows[
                0
            ]
        )


        # Force this region into human review.
        result_df.at[
            row_index,
            "potential_missing"
        ] = True


        result_df.at[
            row_index,
            "rule_flag"
        ] = True


        result_df.at[
            row_index,
            "structural_missing_section"
        ] = True


        current_score = float(
            result_df.at[
                row_index,
                "review_score"
            ]
        )


        result_df.at[
            row_index,
            "review_score"
        ] = max(
            current_score,
            0.95
        )


        previous_sections = str(
            result_df.at[
                row_index,
                "missing_section_numbers"
            ]
        ).strip()


        if previous_sections:

            new_value = (
                previous_sections
                + ", "
                + str(
                    section_number
                )
            )

        else:

            new_value = str(
                section_number
            )


        result_df.at[
            row_index,
            "missing_section_numbers"
        ] = new_value


    # --------------------------------------------------------
    # DEBUG OUTPUT
    # --------------------------------------------------------

    print(
        f"\n===== {target_language} SECTION CHECK ====="
    )

    print(
        "Anchor language:",
        anchor_language
    )

    print(
        "Reference language:",
        reference_language
    )

    print(
        "Anchor sections:",
        sorted(
            anchor_sections.keys()
        )
    )

    print(
        "Reference sections:",
        sorted(
            reference_sections.keys()
        )
    )

    print(
        "Target sections:",
        sorted(
            target_sections.keys()
        )
    )

    print(
        "Missing sections:",
        missing_sections
    )

    print(
        "=========================================\n"
    )


    return (
        result_df,
        missing_sections
    )
# ============================================================
# REVIEW DISPLAY
# ============================================================

def display_flagged_sections(
    analysis_df,
    prepared,
    target_language,
    max_items=5
):
    """Display review regions without inventing a target match.

    For a structurally missing numbered section there is, by
    definition, no corresponding target section. In that case the
    UI explicitly reports the missing section instead of displaying
    an unrelated DTW-nearest target chunk.
    """

    flagged_df = (
        analysis_df[
            analysis_df["potential_missing"]
        ]
        .sort_values(
            "review_score",
            ascending=False
        )
        .copy()
    )

    if flagged_df.empty:
        return

    anchor_language = (
        LANGUAGE_CONFIGS[target_language]["anchor"]
    )

    st.markdown(
        f"### {target_language} - Sections to Review"
    )

    for number, (_, row) in enumerate(
        flagged_df.head(max_items).iterrows(),
        start=1
    ):
        source_index = int(row["source_chunk"])
        target_index = int(row["target_chunk"])
        review_score = float(row["review_score"])

        is_structural = bool(
            row.get("structural_missing_section", False)
        )

        missing_sections = str(
            row.get("missing_section_numbers", "")
        ).strip()

        with st.expander(
            f"Review Region {number} "
            f"- Review score: {review_score:.2f}"
        ):
            left_col, right_col = st.columns(2)

            with left_col:
                st.markdown(
                    f"**{anchor_language} reference section**"
                )
                st.write(
                    prepared[anchor_language]["chunks"][
                        source_index
                    ]
                )

            with right_col:
                if is_structural:
                    st.markdown(
                        f"**{target_language} document status**"
                    )

                    if missing_sections:
                        st.warning(
                            f"Numbered section {missing_sections} "
                            f"was not detected in the "
                            f"{target_language} document. "
                            "Manual comparison is recommended."
                        )
                    else:
                        st.warning(
                            f"A corresponding numbered section "
                            f"was not detected in the "
                            f"{target_language} document."
                        )

                    st.caption(
                        "The system does not show an unrelated "
                        "nearest-matching chunk for a section that "
                        "is structurally absent."
                    )

                else:
                    st.markdown(
                        f"**{target_language} matched section**"
                    )

                    target_chunks = (
                        prepared[target_language]["chunks"]
                    )

                    if 0 <= target_index < len(target_chunks):
                        st.write(target_chunks[target_index])
                    else:
                        st.write(
                            "No corresponding section was identified."
                        )



# ============================================================
# UI
# ============================================================

st.title(
    "🔎 TriCheck-LK",
    anchor=False
)


if not TESSERACT_AVAILABLE:

    st.warning(
        "Tesseract OCR was not found. "
        "Direct-text PDFs can still be processed, "
        "but scanned Sinhala/Tamil PDFs may not work. "
        "Install Tesseract with eng, sin and tam "
        "language packs or add Tesseract to PATH."
    )


if (
    "upload_version"
    not in st.session_state
):

    st.session_state.upload_version = 0


st.write(
    "Upload the English, Sinhala and Tamil versions "
    "of the same document to check their "
    "cross-lingual semantic consistency."
)


(
    col1,
    col2,
    col3
) = st.columns(
    3
)


with col1:

    st.markdown(
        "**English PDF**"
    )

    with st.container(
        border=True
    ):

        english_file = (
            st.file_uploader(
                "Upload English PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key=(
                    f"english_pdf_"
                    f"{st.session_state.upload_version}"
                )
            )
        )


with col2:

    st.markdown(
        "**Sinhala PDF**"
    )

    with st.container(
        border=True
    ):

        sinhala_file = (
            st.file_uploader(
                "Upload Sinhala PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key=(
                    f"sinhala_pdf_"
                    f"{st.session_state.upload_version}"
                )
            )
        )


with col3:

    st.markdown(
        "**Tamil PDF**"
    )

    with st.container(
        border=True
    ):

        tamil_file = (
            st.file_uploader(
                "Upload Tamil PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key=(
                    f"tamil_pdf_"
                    f"{st.session_state.upload_version}"
                )
            )
        )


all_files_uploaded = all(
    file is not None
    for file in (
        english_file,
        sinhala_file,
        tamil_file
    )
)


if not all_files_uploaded:

    st.info(
        "Please upload all three corresponding language versions."
    )


(
    button_col1,
    button_col2
) = st.columns(
    [
        1,
        1
    ]
)


with button_col1:

    check_button = (
        st.button(
            "Check Documents",
            type="primary",
            disabled=not all_files_uploaded,
            use_container_width=True
        )
    )


with button_col2:

    reset_button = (
        st.button(
            "Reset Documents",
            type="secondary",
            use_container_width=True
        )
    )


if reset_button:

    st.session_state.upload_version += 1

    st.rerun()


# ============================================================
# MAIN PIPELINE
# ============================================================

if check_button:

    # --------------------------------------------------------
    # STEP 1 - PREPARE
    # --------------------------------------------------------

    with st.spinner(
        "Preparing the three documents..."
    ):

        english_result = (
            extract_pdf_text(
                english_file,
                "English"
            )
        )

        sinhala_result = (
            extract_pdf_text(
                sinhala_file,
                "Sinhala"
            )
        )

        tamil_result = (
            extract_pdf_text(
                tamil_file,
                "Tamil"
            )
        )

        english_chunks = (
            create_chunks(
                english_result[
                    "text"
                ]
            )
        )

        sinhala_chunks = (
            create_chunks(
                sinhala_result[
                    "text"
                ]
            )
        )

        tamil_chunks = (
            create_chunks(
                tamil_result[
                    "text"
                ]
            )
        )


    # --------------------------------------------------------
    # STEP 2 - LANGUAGE VALIDATION
    # --------------------------------------------------------

    validation_results = [

        validate_document(
            english_result,
            english_chunks,
            "English"
        ),

        validate_document(
            sinhala_result,
            sinhala_chunks,
            "Sinhala"
        ),

        validate_document(
            tamil_result,
            tamil_chunks,
            "Tamil"
        ),
    ]

    validation_errors = [
        message
        for (
            valid,
            message
        ) in validation_results
        if not valid
    ]

    if validation_errors:

        for message in (
            validation_errors
        ):

            st.error(
                message
            )

        st.info(
            "Document correspondence checking will continue "
            "after all files are uploaded in their correct "
            "English, Sinhala and Tamil sections."
        )

        st.stop()


    # --------------------------------------------------------
    # STEP 3 - PAIR CHECKING
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Verifying corresponding documents..."
        ):

            prepared = {

                "English": {
                    "chunks":
                        english_chunks,

                    "embeddings":
                        create_embeddings(
                            english_chunks,
                            embedding_model
                        ),

                    "filename":
                        english_file.name,
                },

                "Sinhala": {
                    "chunks":
                        sinhala_chunks,

                    "embeddings":
                        create_embeddings(
                            sinhala_chunks,
                            embedding_model
                        ),

                    "filename":
                        sinhala_file.name,
                },

                "Tamil": {
                    "chunks":
                        tamil_chunks,

                    "embeddings":
                        create_embeddings(
                            tamil_chunks,
                            embedding_model
                        ),

                    "filename":
                        tamil_file.name,
                },
            }

            (
                same_document,
                verification
            ) = (
                verify_same_document(
                    prepared
                )
            )

    except Exception as error:

        st.error(
            "The uploaded documents could not be verified. "
            f"Technical details: {error}"
        )

        st.stop()


    # --------------------------------------------------------
    # STEP 4 - STOP IF DIFFERENT
    # --------------------------------------------------------

    if not same_document:

        pair_results = (
            verification[
                "pair_results"
            ]
        )

        mismatched_language = (
            identify_mismatched_language(
                pair_results
            )
        )

        if mismatched_language in {
            "English",
            "Sinhala",
            "Tamil"
        }:

            st.error(
                f"The {mismatched_language} PDF does not appear "
                f"to correspond with the other two language versions."
            )

            st.info(
                f"Please check and upload the correct "
                f"{mismatched_language} version of the same document."
            )

        else:

            st.error(
                "The uploaded files do not appear to be "
                "corresponding English, Sinhala and Tamil "
                "versions of the same document."
            )

            st.info(
                "Two or more uploaded documents may be unrelated. "
                "Please upload the English, Sinhala and Tamil "
                "versions of the same circular or official document."
            )

        st.stop()


    # --------------------------------------------------------
    # STEP 5 - SEMANTIC ANALYSIS
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Analyzing semantic consistency..."
        ):

            english_features = (
                build_document_features(
                    prepared,
                    "English"
                )
            )

            sinhala_features = (
                build_document_features(
                    prepared,
                    "Sinhala"
                )
            )

            tamil_features = (
                build_document_features(
                    prepared,
                    "Tamil"
                )
            )

            (
                english_analysis,
                _
            ) = (
                predict_missing_information(
                    english_features,
                    "English"
                )
            )

            (
                sinhala_analysis,
                _
            ) = (
                predict_missing_information(
                    sinhala_features,
                    "Sinhala"
                )
            )

            (
                tamil_analysis,
                _
            ) = (
                predict_missing_information(
                    tamil_features,
                    "Tamil"
                )
            )
            # ========================================================
            # STRUCTURAL MISSING-SECTION CHECK
            # ========================================================

            (
                english_analysis,
                english_missing_sections
            ) = apply_numbered_section_rule(
                english_analysis,
                prepared,
                "English"
            )
            (
                    sinhala_analysis,
                    sinhala_missing_sections
            ) = apply_numbered_section_rule(
                    sinhala_analysis,
                    prepared,
                    "Sinhala"
            )


            (
                        tamil_analysis,
                        tamil_missing_sections
                    ) = apply_numbered_section_rule(
                        tamil_analysis,
                        prepared,
                        "Tamil"
            )
    except Exception as error:

        st.error(
            "The documents could not be analyzed. "
            f"Technical details: {error}"
        )

        st.stop()


    st.success(
        "Documents analyzed successfully."
    )


    # ========================================================
    # PREPARATION SUMMARY
    # ========================================================

    st.subheader(
        "Document Preparation Summary"
    )

    summary_data = {

        "Language": [
            "English",
            "Sinhala",
            "Tamil"
        ],

        "Detected Language": [
            english_result[
                "detected_language"
            ],

            sinhala_result[
                "detected_language"
            ],

            tamil_result[
                "detected_language"
            ],
        ],

        "Extraction Method": [
            english_result[
                "method"
            ],

            sinhala_result[
                "method"
            ],

            tamil_result[
                "method"
            ],
        ],

        "Characters": [
            english_result[
                "characters"
            ],

            sinhala_result[
                "characters"
            ],

            tamil_result[
                "characters"
            ],
        ],

        "Chunks": [
            len(
                english_chunks
            ),

            len(
                sinhala_chunks
            ),

            len(
                tamil_chunks
            ),
        ],
    }

    st.dataframe(
        summary_data,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # RESULTS
    # ========================================================

    english_flagged = int(
        english_analysis[
            "potential_missing"
        ]
        .sum()
    )

    sinhala_flagged = int(
        sinhala_analysis[
            "potential_missing"
        ]
        .sum()
    )

    tamil_flagged = int(
        tamil_analysis[
            "potential_missing"
        ]
        .sum()
    )

    total_flagged = (
        english_flagged
        +
        sinhala_flagged
        +
        tamil_flagged
    )

    st.subheader(
        "Document Check Result"
    )

    if total_flagged == 0:

        st.success(
            "No potential missing information was "
            "detected across the three language versions."
        )

        st.caption(
            "TriCheck-LK is a decision-support system. "
            "Important documents should still be "
            "reviewed manually when necessary."
        )

    else:

        st.warning(
            "Some sections may require manual review."
        )


    result_table = {

        "Language": [
            "English",
            "Sinhala",
            "Tamil"
        ],

        "Flagged Review Regions": [
            english_flagged,
            sinhala_flagged,
            tamil_flagged,
        ],

        "Status": [

            (
                "No issue detected"
                if english_flagged
                == 0
                else
                "Review recommended"
            ),

            (
                "No issue detected"
                if sinhala_flagged
                == 0
                else
                "Review recommended"
            ),

            (
                "No issue detected"
                if tamil_flagged
                == 0
                else
                "Review recommended"
            ),
        ],
    }

    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # FLAGGED SECTIONS
    # ========================================================

    if total_flagged > 0:

        st.divider()

        st.subheader(
            "Sections Recommended for Review"
        )

        st.write(
           "The following sections were identified for manual comparison."
        )

        if english_flagged > 0:

            display_flagged_sections(
                english_analysis,
                prepared,
                "English"
            )

        if sinhala_flagged > 0:

            display_flagged_sections(
                sinhala_analysis,
                prepared,
                "Sinhala"
            )

        if tamil_flagged > 0:

            display_flagged_sections(
                tamil_analysis,
                prepared,
                "Tamil"
            )