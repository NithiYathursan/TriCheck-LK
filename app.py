
# TriCheck-LK
# Cross-Lingual Semantic Consistency and Missing-Information Detection

from pathlib import Path
import io
import json
import re
import unicodedata
import joblib
import numpy as np
import pandas as pd
import pymupdf
import pytesseract
import streamlit as st

from PIL import Image
from sentence_transformers import SentenceTransformer

# PAGE / PATH CONFIGURATION

st.set_page_config(
    page_title="TriCheck-LK",
    page_icon="🔎",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"

# TESSERACT CONFIGURATION

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# GENERAL CONFIGURATION


MIN_TEXT_CHARS = 500
MAX_CHUNK_CHARS = 1000

LANGUAGE_OCR_CODES = {
    "English": "eng",
    "Sinhala": "sin",
    "Tamil": "tam",
}

LANGUAGE_CONFIGS = {
    "English": {
        "anchor": "Sinhala",
        "reference": "Tamil"
    },

    "Sinhala": {
        "anchor": "English",
        "reference": "Tamil"
    },

    "Tamil": {
        "anchor": "English",
        "reference": "Sinhala"
    },
}

# LOAD MODELS


@st.cache_resource
def load_resources():

    detectors = {
        "English": joblib.load(
            MODEL_DIR
            / "tricheck_english_detector.joblib"
        ),

        "Sinhala": joblib.load(
            MODEL_DIR
            / "tricheck_sinhala_detector.joblib"
        ),

        "Tamil": joblib.load(
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

        model_config = json.load(
            file
        )


    embedding_model = SentenceTransformer(
        model_config[
            "embedding_model"
        ]
    )


    return (
        detectors,
        model_config,
        embedding_model
    )


detectors, model_config, embedding_model = (
    load_resources()
)


# TEXT CLEANING
# Same preprocessing used during model development


def clean_text(text):

    text = str(
        text
    )


    # Unicode normalization
    text = unicodedata.normalize(
        "NFC",
        text
    )


    # Keep newline, tab, ZWNJ and ZWJ
    text = "".join(
        char
        for char in text
        if (
            unicodedata.category(char)[0]
            != "C"
            or char in "\n\t\u200c\u200d"
        )
    )


    # Tabs -> spaces
    text = text.replace(
        "\t",
        " "
    )


    # Collapse repeated spaces
    text = re.sub(
        r"[ ]+",
        " ",
        text
    )


    # Collapse excessive blank lines
    text = re.sub(
        r"\n\s*\n+",
        "\n",
        text
    )


    return text.strip()



# DIRECT PDF TEXT EXTRACTION

def extract_direct_text(
    pdf_bytes
):

    try:

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )


        extracted_text = ""


        for page in document:

            extracted_text += (
                page.get_text(
                    "text"
                )
            )

            extracted_text += "\n"


        document.close()


        extracted_text = (
            extracted_text.strip()
        )


        if (
            len(extracted_text)
            >= MIN_TEXT_CHARS
        ):

            return (
                extracted_text,
                "DIRECT_TEXT"
            )


        if (
            len(extracted_text)
            > 0
        ):

            return (
                extracted_text,
                "LOW_TEXT"
            )


        return (
            extracted_text,
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

# OCR EXTRACTION

def extract_ocr_text(
    pdf_bytes,
    language
):

    try:

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )


        ocr_text = ""


        language_code = (
            LANGUAGE_OCR_CODES[
                language
            ]
        )


        for page in document:

            # Same 300 DPI used during dataset extraction
            pixmap = page.get_pixmap(
                dpi=300
            )


            image_bytes = (
                pixmap.tobytes(
                    "png"
                )
            )


            image = Image.open(
                io.BytesIO(
                    image_bytes
                )
            )


            page_text = (
                pytesseract.image_to_string(
                    image,
                    lang=language_code,
                    config="--oem 3 --psm 3"
                )
            )


            ocr_text += page_text
            ocr_text += "\n"


        document.close()


        return (
            ocr_text.strip()
        )


    except Exception as error:

        print(
            "OCR extraction error:",
            error
        )

        return ""

# HYBRID PDF EXTRACTION

def extract_pdf_text(
    uploaded_file,
    language
):

    pdf_bytes = (
        uploaded_file.getvalue()
    )


    # First try direct extraction
    raw_text, status = (
        extract_direct_text(
            pdf_bytes
        )
    )


    if (
        status
        == "DIRECT_TEXT"
    ):

        cleaned_text = (
            clean_text(
                raw_text
            )
        )


        return {
            "text":
                cleaned_text,

            "method":
                "DIRECT_TEXT",

            "raw_characters":
                len(raw_text),

            "characters":
                len(cleaned_text),
        }


    # OCR fallback
    if status in {
        "LOW_TEXT",
        "EMPTY"
    }:

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


        final_status = (
            "OCR_TEXT"
            if (
                len(raw_ocr_text)
                >= MIN_TEXT_CHARS
            )
            else "UNUSABLE"
        )


        return {
            "text":
                cleaned_ocr_text,

            "method":
                final_status,

            "raw_characters":
                len(raw_ocr_text),

            "characters":
                len(cleaned_ocr_text),
        }


    return {
        "text": "",
        "method": "ERROR",
        "raw_characters": 0,
        "characters": 0,
    }

# TEXT CHUNKING
# Exact logic used during model development

def create_chunks(
    text,
    max_chars=MAX_CHUNK_CHARS
):

    # Convert repeated whitespace and line breaks
    # into normal spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    # Sentence splitting
    raw_sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    # Join standalone section numbers

    sentences = []

    index = 0


    while (
        index
        < len(raw_sentences)
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
            < len(raw_sentences)
        ):

            next_sentence = (
                raw_sentences[
                    index + 1
                ]
                .strip()
            )


            sentences.append(
                current
                + " "
                + next_sentence
            )


            index += 2


        else:

            if current:

                sentences.append(
                    current
                )


            index += 1

    # Build chunks

    chunks = []

    current_chunk = ""


    for sentence in sentences:

        sentence = (
            sentence.strip()
        )


        if not sentence:

            continue


        # Long OCR/punctuation-less sentence
        if (
            len(sentence)
            > max_chars
        ):

            for word in sentence.split():

                if (
                    len(current_chunk)
                    + len(word)
                    + 1
                    <= max_chars
                ):

                    if current_chunk:

                        current_chunk += " "


                    current_chunk += word


                else:

                    if current_chunk:

                        chunks.append(
                            current_chunk.strip()
                        )


                    current_chunk = word


            continue


        # Normal sentence
        if (
            len(current_chunk)
            + len(sentence)
            + 1
            <= max_chars
        ):

            if current_chunk:

                current_chunk += " "


            current_chunk += sentence


        else:

            if current_chunk:

                chunks.append(
                    current_chunk.strip()
                )


            current_chunk = sentence


    if current_chunk:

        chunks.append(
            current_chunk.strip()
        )


    return chunks

# LANGUAGE SCRIPT VALIDATION

def detect_text_language(text):

    english_count = 0
    sinhala_count = 0
    tamil_count = 0

    for char in text:

        code = ord(char)

        # English / Latin letters
        if (
            "A" <= char <= "Z"
            or "a" <= char <= "z"
        ):
            english_count += 1

        # Sinhala Unicode range
        elif 0x0D80 <= code <= 0x0DFF:
            sinhala_count += 1

        # Tamil Unicode range
        elif 0x0B80 <= code <= 0x0BFF:
            tamil_count += 1


    counts = {
        "English": english_count,
        "Sinhala": sinhala_count,
        "Tamil": tamil_count
    }

    detected_language = max(
        counts,
        key=counts.get
    )

    total_letters = sum(
        counts.values()
    )

    if total_letters == 0:
        return "Unknown"

    dominance = (
        counts[detected_language]
        / total_letters
    )

    if dominance < 0.60:
        return "Unknown"

    return detected_language

# DOCUMENT VALIDATION


def validate_document(
    result,
    chunks,
    language
):


    # Extraction error

    if result["method"] == "ERROR":

        return (
            False,
            f"{language} PDF could not be processed."
        )

    # Insufficient OCR/direct text


    if result["method"] == "UNUSABLE":

        return (
            False,
            f"{language} PDF does not contain enough "
            "usable text for analysis."
        )

    # Empty text

    if not result["text"].strip():

        return (
            False,
            f"No usable text was extracted from the "
            f"{language} PDF."
        )

    # No chunks

    if len(chunks) == 0:

        return (
            False,
            f"No text chunks could be created from the "
            f"{language} PDF."
        )

    # Language-slot validation

    detected_language = detect_text_language(
        result["text"]
    )


    if (
        detected_language != "Unknown"
        and
        detected_language != language
    ):

        return (
            False,
            f"The file uploaded in the {language} section "
            f"appears to be a {detected_language} document. "
            f"Please upload the correct {language} version."
        )

    # Document is valid

    return (
        True,
        ""
    )

# MULTILINGUAL EMBEDDINGS

def create_embeddings(
    chunks,
    model
):

    model_inputs = [
        "query: " + chunk
        for chunk in chunks
    ]


    embeddings = model.encode(
        model_inputs,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )


    return embeddings


# DTW SEMANTIC ALIGNMENT

def dtw_align(
    source_embeddings,
    target_embeddings
):

    # Embeddings are normalized.
    # Dot product therefore gives cosine similarity.
    similarity_matrix = np.matmul(
        source_embeddings,
        target_embeddings.T
    )


    cost_matrix = (
        1.0
        - similarity_matrix
    )


    source_count = (
        cost_matrix.shape[0]
    )

    target_count = (
        cost_matrix.shape[1]
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

    # Fill DTW table

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


    # Backtrack

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
            ]
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


    # Remaining source positions
    while (
        i > 0
    ):

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


    # Remaining target positions
    while (
        j > 0
    ):

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

# SOURCE COVERAGE

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


    best_scores = []


    for source_pos in range(
        source_count
    ):

        scores = (
            coverage[
                source_pos
            ]
        )


        if scores:

            best_scores.append(
                max(scores)
            )


        else:

            best_scores.append(
                np.nan
            )


    return best_scores


# ALIGNMENT PATH DATAFRAME

def create_path_dataframe(
    path
):

    rows = []


    for (
        source_pos,
        target_pos,
        similarity
    ) in path:

        rows.append(
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
        )


    return pd.DataFrame(
        rows
    )


# ALIGNMENT COMPRESSION FEATURE


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


# REAL-DOCUMENT FEATURE GENERATION


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
        ]["chunks"]
    )


    target_chunks = (
        prepared[
            target_language
        ]["chunks"]
    )


    anchor_embeddings = (
        prepared[
            anchor_language
        ]["embeddings"]
    )


    target_embeddings = (
        prepared[
            target_language
        ]["embeddings"]
    )


    reference_embeddings = (
        prepared[
            reference_language
        ]["embeddings"]
    )

    # Anchor <-> target alignment

    target_path, _ = dtw_align(
        anchor_embeddings,
        target_embeddings
    )

    # Anchor <-> reference alignment


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

    # Alignment path features

    path_df = (
        create_path_dataframe(
            target_path
        )
    )


    path_df = (
        add_compression_feature(
            path_df
        )
    )

    # Position difference

    source_max = max(
        len(anchor_chunks)
        - 1,
        1
    )


    target_max = max(
        len(target_chunks)
        - 1,
        1
    )


    path_df[
        "position_difference"
    ] = abs(
        (
            path_df[
                "source_position"
            ]
            / source_max
        )
        -
        (
            path_df[
                "target_position"
            ]
            / target_max
        )
    )

    # Aggregate target features
  

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
            )
        )
        .reset_index()
    )

    # Reference-language similarity
  

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

    # Local compression
   

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


    
    # Cross-language disagreement
   

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


    
    # Local semantic anomaly
   

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


   
    # Best target chunk for side-by-side display
   

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


# LANGUAGE-SPECIFIC MODEL PREDICTION


def predict_missing_information(
    feature_df,
    language
):

    language_model_config = (
        model_config[
            "languages"
        ][
            language
        ]
    )


    feature_columns = (
        language_model_config[
            "features"
        ]
    )


    threshold = float(
        language_model_config[
            "threshold"
        ]
    )


    # Exact training feature order
    X = (
        feature_df[
            feature_columns
        ]
        .copy()
    )


    # Safety check
    if (
        X.isna()
        .any()
        .any()
    ):

        raise ValueError(
            f"{language} features "
            "contain missing values."
        )


    detector = (
        detectors[
            language
        ]
    )


    probabilities = (
        detector
        .predict_proba(
            X
        )[:, 1]
    )


    predictions = (
        probabilities
        >= threshold
    )


    result_df = (
        feature_df.copy()
    )


    result_df[
        "missing_probability"
    ] = probabilities


    result_df[
        "potential_missing"
    ] = predictions


    return (
        result_df,
        threshold
    )



# FLAGGED SECTION DISPLAY


def display_flagged_sections(
    analysis_df,
    prepared,
    target_language,
    max_items=5
):

    flagged_df = (
        analysis_df[
            analysis_df[
                "potential_missing"
            ]
        ]
        .sort_values(
            "missing_probability",
            ascending=False
        )
        .copy()
    )


    if (
        flagged_df.empty
    ):

        return


    anchor_language = (
        LANGUAGE_CONFIGS[
            target_language
        ][
            "anchor"
        ]
    )


    st.markdown(
        f"### {target_language} - "
        "Sections to Review"
    )


    for (
        number,
        (_, row)
    ) in enumerate(
        flagged_df
        .head(
            max_items
        )
        .iterrows(),
        start=1
    ):

        source_index = int(
            row[
                "source_chunk"
            ]
        )


        target_index = int(
            row[
                "target_chunk"
            ]
        )


        model_score = float(
            row[
                "missing_probability"
            ]
        )


        with st.expander(
            f"Review Region {number} "
            f"- Model score: "
            f"{model_score:.2f}"
        ):

            left_col, right_col = (
                st.columns(
                    2
                )
            )


            with left_col:

                st.markdown(
                    f"**{anchor_language} "
                    "reference section**"
                )


                st.write(
                    prepared[
                        anchor_language
                    ][
                        "chunks"
                    ][
                        source_index
                    ]
                )


            with right_col:

                st.markdown(
                    f"**{target_language} "
                    "matched section**"
                )


                if (
                    0
                    <= target_index
                    < len(
                        prepared[
                            target_language
                        ][
                            "chunks"
                        ]
                    )
                ):

                    st.write(
                        prepared[
                            target_language
                        ][
                            "chunks"
                        ][
                            target_index
                        ]
                    )


                else:

                    st.write(
                        "No corresponding section "
                        "was identified."
                    )



# MAIN APPLICATION UI


st.title(
    "🔎 TriCheck-LK"
)

# UPLOAD RESET STATE


if "upload_version" not in st.session_state:
    st.session_state.upload_version = 0

st.write(
    "Upload the English, Sinhala and Tamil versions "
    "of the same document to check their "
    "cross-lingual semantic consistency."
)


col1, col2, col3 = (
    st.columns(
        3
    )
)


with col1:

    english_file = (
        st.file_uploader(
            "English PDF",
            type=["pdf"],
            key=f"english_pdf_{st.session_state.upload_version}"
        )
    )


with col2:

    sinhala_file = (
        st.file_uploader(
            "Sinhala PDF",
            type=["pdf"],
            key=f"sinhala_pdf_{st.session_state.upload_version}"
        )
    )


with col3:

    tamil_file = (
        st.file_uploader(
            "Tamil PDF",
            type=["pdf"],
            key=f"tamil_pdf_{st.session_state.upload_version}"
        )
    )


all_files_uploaded = (
    english_file
    is not None
    and
    sinhala_file
    is not None
    and
    tamil_file
    is not None
)


if not all_files_uploaded:

    st.info(
        "Please upload all three corresponding "
        "language versions."
    )

button_col1, button_col2 = st.columns(
    [1, 1]
)


with button_col1:

    check_button = st.button(
        "Check Documents",
        type="primary",
        disabled=not all_files_uploaded,
        use_container_width=True
    )


with button_col2:

    reset_button = st.button(
        "Reset Documents",
        use_container_width=True
    )


if reset_button:

    st.session_state.upload_version += 1

    st.rerun()

# CHECK DOCUMENTS


if check_button:

    
    # STEP 1 - PREPARE DOCUMENTS
    

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


    # STEP 2 - VALIDATE DOCUMENTS
   

    english_valid, english_error = (
        validate_document(
            english_result,
            english_chunks,
            "English"
        )
    )


    sinhala_valid, sinhala_error = (
        validate_document(
            sinhala_result,
            sinhala_chunks,
            "Sinhala"
        )
    )


    tamil_valid, tamil_error = (
        validate_document(
            tamil_result,
            tamil_chunks,
            "Tamil"
        )
    )


    validation_errors = [
        message

        for valid, message in [
            (
                english_valid,
                english_error
            ),

            (
                sinhala_valid,
                sinhala_error
            ),

            (
                tamil_valid,
                tamil_error
            ),
        ]

        if not valid
    ]


    if validation_errors:

        for message in validation_errors:

            st.error(
                message
            )


        st.stop()


    # STEP 3 - NLP ANALYSIS
    

    try:

        with st.spinner(
            "Analyzing semantic consistency..."
        ):

            english_embeddings = (
                create_embeddings(
                    english_chunks,
                    embedding_model
                )
            )


            sinhala_embeddings = (
                create_embeddings(
                    sinhala_chunks,
                    embedding_model
                )
            )


            tamil_embeddings = (
                create_embeddings(
                    tamil_chunks,
                    embedding_model
                )
            )


            prepared = {

                "English": {
                    "chunks":
                        english_chunks,

                    "embeddings":
                        english_embeddings,
                },


                "Sinhala": {
                    "chunks":
                        sinhala_chunks,

                    "embeddings":
                        sinhala_embeddings,
                },


                "Tamil": {
                    "chunks":
                        tamil_chunks,

                    "embeddings":
                        tamil_embeddings,
                },
            }


         
            # Feature generation
           

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


            
            # Predictions
           

            english_analysis, _ = (
                predict_missing_information(
                    english_features,
                    "English"
                )
            )


            sinhala_analysis, _ = (
                predict_missing_information(
                    sinhala_features,
                    "Sinhala"
                )
            )


            tamil_analysis, _ = (
                predict_missing_information(
                    tamil_features,
                    "Tamil"
                )
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


  
    # DOCUMENT PREPARATION SUMMARY
   

    st.subheader(
        "Document Preparation Summary"
    )


    summary_data = {

        "Language": [
            "English",
            "Sinhala",
            "Tamil"
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


    
    # FINAL DOCUMENT CHECK RESULT
  

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


    if (
        total_flagged
        == 0
    ):

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
                if (
                    english_flagged
                    == 0
                )
                else
                "Review recommended"
            ),

            (
                "No issue detected"
                if (
                    sinhala_flagged
                    == 0
                )
                else
                "Review recommended"
            ),

            (
                "No issue detected"
                if (
                    tamil_flagged
                    == 0
                )
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


  
    # POTENTIAL SECTIONS FOR MANUAL REVIEW
    

    if (
        total_flagged
        > 0
    ):

        st.divider()


        st.subheader(
            "Sections Recommended for Review"
        )


        st.write(
            "The following sections were flagged "
            "by the model for manual comparison."
        )


        if (
            english_flagged
            > 0
        ):

            display_flagged_sections(
                english_analysis,
                prepared,
                "English"
            )


        if (
            sinhala_flagged
            > 0
        ):

            display_flagged_sections(
                sinhala_analysis,
                prepared,
                "Sinhala"
            )


        if (
            tamil_flagged
            > 0
        ):

            display_flagged_sections(
                tamil_analysis,
                prepared,
                "Tamil"
            )