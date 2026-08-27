import csv
import requests

#Metadata containing only usable trilingual records
candidate_file="data/metadata/public_admin_nlp_candidates.csv"

#Read candidate records
with open(candidate_file,"r",encoding="utf-8-sig") as file:
    reader=csv.DictReader(file)
    rows=list(reader)

print("Candidate records:",len(rows))

#Test only the first 10 circulars
test_rows=rows[:10]

language_columns=["english_url","sinhala_url","tamil_url"]

valid_pdfs = 0
invalid_pdfs = 0

for row in test_rows:
    print("\nCircular:",row["circular_number"])
    for column in language_columns:
        url=row[column]

        try:
            #stream=True avoids downloading the whole PDF

            response=requests.get(url,stream=True,timeout=30)
            response.raise_for_status()

            #PDF files normally begin with the bytes %PDF
            first_bytes=next(response.iter_content(chunk_size=4),b"")

            if first_bytes.startswith(b"%PDF"):
                print(column,"-> VALID PDF")
                valid_pdfs += 1

            else:
                print(column,"-> NOT A PDF")
                invalid_pdfs +=1

            response.close()    

        except requests.RequestException as error:
            print(column,"-> FAILED:",error)
            invalid +=1

print("\nValid PDFs:",valid_pdfs)
print("Invalid PDFs:",invalid_pdfs)                    
