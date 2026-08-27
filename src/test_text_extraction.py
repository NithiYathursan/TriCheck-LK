import csv
import requests
import pymupdf
import io

candidate_file="data/metadata/public_admin_nlp_candidates.csv"

with open(candidate_file,"r",encoding="utf-8-sig") as file:
    reader=csv.DictReader(file)
    rows=list(reader)

#Test only first usable circular
test_row = rows[0]

print("Circular:",test_row["circular_number"])

#Get the English PDF URL
english_url=test_row["english_url"]

#Download the PDF
response = requests.get(english_url,timeout=30)

response.raise_for_status()

#Open the PDF directly from downloaded bytes
document = pymupdf.open(stream=response.content,filetype="pdf")

print("English PDF pages:",len(document))

#Extract text from the first 2 pages
english_text = ""

for page_number in range(min(2,len(document))):
    page=document[page_number]

    #Extract readable text from the PDF page
    page_text=page.get_text("text")

    english_text += page_text


print("\nExtracted English Characters:",len(english_text))
print("\nEnglish Text Sample:")   

print("*"*50)

#Print only the first 1500 characters
print(english_text[:1500])

document.close()

def test_language_pdf(pdf_url,language_name):
    #Download the PDF
    response = requests.get(pdf_url,timeout=30)
    response.raise_for_status()

    #Open the PDF directly from memory
    document=pymupdf.open(stream=response.content,filetype="pdf")

    print(f"\n{language_name} PDF pages",len(document))

    extracted_text=""

    #Extracted only first 2 pages for testing
    for page_number in range(min(2,len(document))):
        page = document[page_number]
        extracted_text += page.get_text("text")

    print(f"{language_name} extracted characters:",len(extracted_text))

    print(f"\n{language_name} text sample:")

    print("-"*45)

    #Display only first 1000 characters
    print(extracted_text[:1000])

    document.close()


test_language_pdf(test_row["sinhala_url"],"Sinhala")
test_language_pdf(test_row["tamil_url"],"Tamil")