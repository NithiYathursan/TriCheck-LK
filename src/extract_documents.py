
# TriCheck-LK
# Multilingual Government Circular Text Extraction

import csv
import io
import json
import os
import pymupdf
import pytesseract
import requests
from PIL import Image

# TESSERACT CONFIGURATION


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# EXTRACTION CONFIGURATION


# Minimum extracted text length required
# to accept direct PDF text extraction.
MIN_TEXT_CHARS = 500


# Metadata URL column -> language information
LANGUAGE_CONFIG = {

    "english_url": {
        "name": "English",
        "code": "eng"
    },

    "sinhala_url": {
        "name": "Sinhala",
        "code": "sin"
    },

    "tamil_url": {
        "name": "Tamil",
        "code": "tam"
    }
}


# Known problematic circular sets.
EXCLUDED_CIRCULARS = {
    "12/2014",
    "27/2015"
}

# DIRECT PDF TEXT EXTRACTION


def extract_direct_text(pdf_url):

    """
    Download a PDF and attempt normal text extraction
    using PyMuPDF.

    Returns:
        text, extraction_status
    """

    try:

        # Download PDF
        response = requests.get(
            pdf_url,
            timeout=30
        )

        response.raise_for_status()


        # Open PDF directly from downloaded bytes
        document = pymupdf.open(
            stream=response.content,
            filetype="pdf"
        )


        extracted_text = ""


        # Extract text from every page
        for page in document:

            extracted_text += page.get_text(
                "text"
            )

            extracted_text += "\n"


        document.close()


        # Remove leading/trailing whitespace only.
        # Further preprocessing is performed later
        # in the preprocessing pipeline.
        extracted_text = (
            extracted_text.strip()
        )


        # Enough text was extracted
        if (
            len(extracted_text)
            >= MIN_TEXT_CHARS
        ):

            return (
                extracted_text,
                "DIRECT_TEXT"
            )


        # Some text exists but is insufficient
        elif len(extracted_text) > 0:

            return (
                extracted_text,
                "LOW_TEXT"
            )


        # PDF opened successfully but contains
        # no extractable text
        else:

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
    pdf_url,
    language_code
):

    """
    Extract text from a PDF using Tesseract OCR.

    Used when direct extraction is empty or
    contains insufficient text.
    """

    try:

        # Download PDF
        response = requests.get(
            pdf_url,
            timeout=30
        )

        response.raise_for_status()


        # Open PDF
        document = pymupdf.open(
            stream=response.content,
            filetype="pdf"
        )


        ocr_text = ""


        # OCR every page
        for page in document:

            # Render page at high resolution
            pixmap = page.get_pixmap(
                dpi=300
            )


            # Convert rendered page to PNG bytes
            image_bytes = pixmap.tobytes(
                "png"
            )


            # Open as PIL image
            image = Image.open(
                io.BytesIO(
                    image_bytes
                )
            )


            # Perform OCR
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


        ocr_text = (
            ocr_text.strip()
        )


        return ocr_text


    except Exception as error:

        print(
            "OCR extraction error:",
            error
        )

        return ""



# HYBRID EXTRACTION


def extract_text_with_fallback(
    pdf_url,
    language_code
):

    """
    Hybrid extraction strategy:

    1. Try direct PDF text extraction.
    2. If direct text is unavailable or too short,
       use OCR.
    """

    # Attempt direct extraction first
    

    text, status = extract_direct_text(
        pdf_url
    )


    # Direct extraction successful
    if status == "DIRECT_TEXT":

        return (
            text,
            "DIRECT_TEXT"
        )


    # OCR fallback
    

    if status in [
        "EMPTY",
        "LOW_TEXT"
    ]:

        print(
            "Direct text unavailable. "
            "Trying OCR..."
        )


        ocr_text = extract_ocr_text(
            pdf_url,
            language_code
        )


        # OCR produced sufficient text
        if (
            len(ocr_text)
            >= MIN_TEXT_CHARS
        ):

            return (
                ocr_text,
                "OCR_TEXT"
            )


        # OCR also insufficient
        return (
            ocr_text,
            "UNUSABLE"
        )


    # Download/opening failure
    return (
        "",
        "ERROR"
    )

# MAIN DATASET EXTRACTION PIPELINE


if __name__ == "__main__":

    # -----------------------------------------
    # File paths
    # -----------------------------------------

    candidate_file = (
        "data/metadata/"
        "public_admin_nlp_candidates.csv"
    )


    output_file = (
        "data/processed/"
        "extracted_documents_full.jsonl"
    )


    # Create output folder if required
    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    # LOAD NLP CANDIDATES


    with open(
        candidate_file,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(
            file
        )

        rows = list(
            reader
        )


    print(
        "Total candidate circulars:",
        len(rows)
    )

    # REMOVE KNOWN PROBLEMATIC CIRCULARS

    selected_rows = [

        row

        for row in rows

        if (
            row["circular_number"]
            not in EXCLUDED_CIRCULARS
        )
    ]


    print(
        "\nSelected circulars:",
        len(selected_rows)
    )


    print(
        "Maximum possible documents:",
        len(selected_rows) * 3
    )

    # EXTRACT ALL THREE LANGUAGES

    extracted_records = []


    for index, row in enumerate(
        selected_rows,
        start=1
    ):

        print(
            "\n" + "=" * 60
        )


        print(
            f"Circular "
            f"{index}/"
            f"{len(selected_rows)}:",
            row["circular_number"]
        )


        # Process English, Sinhala and Tamil
        for (
            url_column,
            language_config
        ) in LANGUAGE_CONFIG.items():

            language_name = (
                language_config[
                    "name"
                ]
            )


            language_code = (
                language_config[
                    "code"
                ]
            )


            pdf_url = (
                row[
                    url_column
                ]
            )


            print(
                f"Processing "
                f"{language_name}..."
            )


            # Hybrid extraction
            text, status = (
                extract_text_with_fallback(
                    pdf_url,
                    language_code
                )
            )


            # Create one record
            # for each language document
            record = {

                "circular_id":
                    row.get(
                        "circular_id",
                        ""
                    ),

                "circular_number":
                    row[
                        "circular_number"
                    ],

                "year":
                    row[
                        "year"
                    ],

                "language":
                    language_name,

                "pdf_url":
                    pdf_url,

                "extraction_status":
                    status,

                "character_count":
                    len(text),

                "text":
                    text
            }


            extracted_records.append(
                record
            )


            print(
                "Status:",
                status,
                "| Characters:",
                len(text)
            )

    # SAVE JSONL DATASET

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        for record in extracted_records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


    # FINAL SUMMARY
    

    print(
        "\n" + "=" * 60
    )


    print(
        "Total extracted documents:",
        len(extracted_records)
    )


    print(
        "Saved to:",
        output_file
    )