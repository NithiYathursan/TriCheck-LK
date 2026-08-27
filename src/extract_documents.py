import requests
import pymupdf
import csv
import io
import pytesseract
from PIL import Image
import json
import os
from collections import Counter
import random

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Minimum amount of text required to consider direct extraction successful

MIN_TEXT_CHARS = 500
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
# Circulars excluded because the three-language
# document set is incomplete or mismatched
EXCLUDED_CIRCULARS = {"12/2014","27/2015"}

def extract_direct_text(pdf_url):

    try:
        # Download the PDF
        response = requests.get( pdf_url, timeout=30 )

        response.raise_for_status()

        # Open PDF from downloaded bytes
        document = pymupdf.open( stream=response.content,filetype="pdf" )
           
        extracted_text = ""

        # Extract text from every page
        for page in document:
            extracted_text += page.get_text("text")
            extracted_text += "\n"

        document.close()

        extracted_text = extracted_text.strip()

        # Enough text was extracted
        if len(extracted_text) >= MIN_TEXT_CHARS:
            return extracted_text, "DIRECT_TEXT"

        # Some text exists, but not enough
        elif len(extracted_text) > 0:
            return extracted_text, "LOW_TEXT"

        # PDF opened, but no text was extracted
        else:
            return extracted_text, "EMPTY"

    except Exception as error:

        print( "Direct extraction error:", error )
        
        return "", "ERROR"

def extract_ocr_text(pdf_url, language_code):

    try:
        # Download the scanned PDF
        response = requests.get(pdf_url, timeout=30 )
            
        response.raise_for_status()

        # Open PDF
        document = pymupdf.open(stream=response.content, filetype="pdf" )
        
        ocr_text = ""

        # OCR every page
        for page in document:

            # Convert page to image
            pixmap = page.get_pixmap(dpi=300 )
                
        
            image_bytes = pixmap.tobytes( "png")
               
            image = Image.open(io.BytesIO(image_bytes))
                
            # Extract text using Tesseract
            page_text = pytesseract.image_to_string(
                image,
                lang=language_code,
                config="--oem 3 --psm 3"
            )

            ocr_text += page_text
            ocr_text += "\n"

        document.close()

        ocr_text = ocr_text.strip()

        return ocr_text

    except Exception as error:

        print( "OCR extraction error:",  error )
           
        return ""

def extract_text_with_fallback( pdf_url,language_code):

    # First try normal PDF text extraction
    text, status = extract_direct_text(pdf_url)
        
    # Good direct text is available
    if status == "DIRECT_TEXT":
        return text, "DIRECT_TEXT"

    # If direct text is missing or too small,
    # try OCR
    if status in ["EMPTY", "LOW_TEXT"]:

        print( "Direct text unavailable. Trying OCR..." )
        
        ocr_text = extract_ocr_text( pdf_url, language_code)
           
        if len(ocr_text) >= MIN_TEXT_CHARS:
            return ocr_text, "OCR_TEXT"

        return ocr_text, "UNUSABLE"

    # Download/PDF error
    return "", "ERROR"

if __name__ == "__main__":

    # Input and output files

    candidate_file = ("data/metadata/public_admin_nlp_candidates.csv" )
       
    output_file = ("data/processed/extracted_documents.jsonl" )
        
    # Create output folder if it does not exist
    os.makedirs( "data/processed",  exist_ok=True )
       
    # Load candidate circulars
   
    with open( candidate_file, "r", encoding="utf-8-sig" ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    print( "Total candidate circulars:", len(rows))
    
    # Keep records from 2012 onwards
    # and remove known problematic circulars

    recent_rows = []

    for row in rows:

        if ( int(row["year"]) >= 2012 and row["circular_number"] not in EXCLUDED_CIRCULARS ):
           
            recent_rows.append(row)

    print( "Candidates from 2012 onwards:",  len(recent_rows)  )
  
    # Check candidate distribution by year

    year_counts = Counter(  row["year"]for row in recent_rows )
    
    print(  "\nCandidate distribution by year:" )
      
    for year in sorted(  year_counts,  reverse=True  ):
        print(
            year,
            "→",
            year_counts[year]
        )

    # Group candidate rows by year

    random.seed(42)

    rows_by_year = {}

    for row in recent_rows:

        year = row["year"]

        if year not in rows_by_year:
            rows_by_year[year] = []

        rows_by_year[year].append(row)


    # Representative sample of 100 circulars

    sample_per_year = {
        "2026": 5,
        "2025": 7,
        "2024": 4,
        "2023": 6,
        "2022": 8,
        "2021": 7,
        "2020": 5,
        "2019": 7,
        "2018": 6,
        "2017": 9,
        "2016": 8,
        "2015": 7,
        "2014": 8,
        "2013": 7,
        "2012": 6
    }


    selected_rows = []

    for year, number_to_select in sample_per_year.items():

        year_sample = random.sample(rows_by_year[year], number_to_select)
            
        selected_rows.extend( year_sample )
          
    print( "\nSelected circulars:", len(selected_rows) )
       
    # Check selected sample distribution

    selected_year_counts = Counter( row["year"] for row in selected_rows )
       
    print("\nSelected sample distribution:" )
        
  

    for year in sorted( selected_year_counts,reverse=True ):
        print(
            year,
            "→",
            selected_year_counts[year]
        )


    # Extract all 3 languages

    extracted_records = []

    for index, row in enumerate(  selected_rows, start=1):
        print(  "\n" + "=" * 60 )
          
        print(
            f"Circular {index}/{len(selected_rows)}:",
            row["circular_number"]
        )


        # Process English, Sinhala and Tamil
        for url_column, config in LANGUAGE_CONFIG.items():

            language_name = config["name"]
            language_code = config["code"]

            pdf_url = row[url_column]

            print(
                f"Processing {language_name}..."
            )


            text, status = extract_text_with_fallback(
                pdf_url,
                language_code
            )


            # One output record per language
            record = {
                "circular_id": row.get(
                    "circular_id",
                    ""
                ),
                "circular_number": row[
                    "circular_number"
                ],
                "year": row["year"],
                "language": language_name,
                "pdf_url": pdf_url,
                "extraction_status": status,
                "character_count": len(text),
                "text": text
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


   
    # Save extracted documents as JSONL


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


    # Final summary

    print( "\n" + "=" * 60 )
    
    print( "Total extracted documents:", len(extracted_records))
       
    print("Saved to:",  output_file   )
        
     
 