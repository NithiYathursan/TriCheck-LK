import csv
import requests
import pymupdf

#Candidate dataset created after metadata validation

candidate_file="data/metadata/public_admin_nlp_candidates.csv"

#Read all NLP candidate records
with open(candidate_file,"r",encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

print("Total candidate records:",len(rows))

#Select 30 records spread across the whole dataset
sample_size = 30

step = len(rows)//sample_size

sample_rows = rows[::step][:sample_size]

print("Selected sample records:",len(sample_rows))

#Language columns available in the candidate dataset
language_columns = {"English":"english_url","Sinhala":"sinhala_url","Tamil":"tamil_url"}

#Store final test counts
results = {
    "English":{"extractable":0,"low_text":0,"needs_ocr":0,"error":0},
    "Sinhala":{"extractable":0,"low_text":0,"needs_ocr":0,"error":0},
    "Tamil":{"extractable":0,"low_text":0,"needs_ocr":0,"error":0}
}

def check_pdf_text(pdf_url):
    try:
        #Download the PDF
        response=requests.get(pdf_url,timeout=30)
        response.raise_for_status()

        #Open PDF directly from downloaded bytes
        document=pymupdf.open(stream=response.content,filetype="pdf")

        extracted_text=""

        #Check only the first 3 pages
        pages_to_check=min(3,len(document))

        for page_number in range(pages_to_check):
            page=document[page_number]
            extracted_text += page.get_text("text")

        document.close()

        #Remove spaces/newlines when counting useful characters
        clean_text = extracted_text.strip()

        character_count=len(clean_text)

        #Classify extraction quality
        if character_count>=100:
            return "extractable",character_count
        elif character_count>0:
            return "low_text",character_count
        else:
            return "needs_ocr",0

    except Exception as error:
        return "error" ,str(error)


print("\nStarting PDF text extractability test..")

for index,row in enumerate(sample_rows,start=1):
    print(f"\nCircular {index}/30 :",row["circular_number"])

    for language,column in language_columns.items():
        status,value=check_pdf_text(row[column])
        results[language][status] +=1

        if status == "error":
            print(language,"-> ERROR:",value)

        else:
            print(language,"->",status.upper(),"| Characters:",value)

print("\n"+"="*50)
print("Text Extractability Summary")
print("-"*50)

for language in results:
    print(f"\n{language}")
    print("Extractable:",results[language]["extractable"])
    print("Low Text:",results[language]["low_text"])
    print("Needs OCR:",results[language]["needs_ocr"])
    print("Errors:",results[language]["error"])


#Count recent candidate circulars
recent_rows =[]

for row in rows:
    year=int(row["year"])
    if year>=2012:
        recent_rows.append(row)

print("\nCandidates from 2012 onwards:",len(recent_rows))        