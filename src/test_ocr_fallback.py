import csv
import io
import requests
import pymupdf
import pytesseract
from PIL import Image


# Tell Python where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")


# --------------------------------------------------
# Function to perform OCR on one PDF page
# --------------------------------------------------
def ocr_pdf_page(page, language_code):

    # Convert PDF page into a high-resolution image
    pixmap = page.get_pixmap(dpi=300)

    # Convert rendered page into PNG bytes
    image_bytes = pixmap.tobytes("png")

    # Open PNG bytes as an image
    image = Image.open(io.BytesIO(image_bytes))

    # Perform OCR using the selected language
    text = pytesseract.image_to_string( image,lang=language_code,config="--oem 3 --psm 3")


    return text


# --------------------------------------------------
# Load NLP candidate dataset
# --------------------------------------------------
candidate_file = ( "data/metadata/public_admin_nlp_candidates.csv")

with open(candidate_file,"r", encoding="utf-8-sig") as file:

    reader = csv.DictReader(file)
    rows = list(reader)


# --------------------------------------------------
# Find circular 24/89
# --------------------------------------------------
test_row = None

for row in rows:

    if row["circular_number"] == "24/89":
        test_row = row
        break


# Stop if the circular was not found
if test_row is None:
    raise ValueError( "Circular 24/89 was not found")

print("Testing circular:",test_row["circular_number"])


# --------------------------------------------------
# Download English PDF
# --------------------------------------------------
english_url = test_row["english_url"]

response = requests.get(english_url,timeout=30)

response.raise_for_status()


# --------------------------------------------------
# Open PDF using PyMuPDF
# --------------------------------------------------
document = pymupdf.open( stream=response.content,filetype="pdf")

print( "English PDF pages:",len(document))


# --------------------------------------------------
# OCR only the first page
# --------------------------------------------------
english_ocr_text = ocr_pdf_page(document[0],"eng")

print( "English OCR characters:",len(english_ocr_text))

print("\nEnglish OCR sample:")
print("*" * 45)

print( english_ocr_text[:1500])

def test_ocr_language(pdf_url, language_name, language_code):

    response = requests.get( pdf_url,timeout=30 )

    response.raise_for_status()

    document = pymupdf.open( stream=response.content, filetype="pdf" )

    print( f"\n{language_name} PDF pages:", len(document))

    # OCR only first page for testing
    ocr_text = ocr_pdf_page( document[0], language_code )

    print( f"{language_name} OCR characters:", len(ocr_text) )

    print( f"\n{language_name} OCR sample:" )

    print("*" * 45)

    print( ocr_text[:1500] )

   
# Close PDF
document.close()

test_ocr_language( test_row["sinhala_url"], "Sinhala", "sin+eng")

test_ocr_language( test_row["tamil_url"], "Tamil", "tam+eng")