import csv
import requests
import pymupdf

candidate_file = ("data/metadata/public_admin_nlp_candidates.csv")
   
problem_circulars = ["17/2022", "12/2014"]

# Load candidate metadata

with open(candidate_file,"r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Inspect known problematic circulars

for circular_number in problem_circulars:

    print( "\n" + "=" * 70 )
       
    print("Circular:", circular_number )
        
    target_row = None

    # Find the required circular
    for row in rows:
        if row["circular_number"] == circular_number:
            target_row = row
            break


    if target_row is None:

        print( "Circular not found" )
        continue


    # URLs for all three languages
    languages = {
        "English": target_row["english_url"],
        "Sinhala": target_row["sinhala_url"],
        "Tamil": target_row["tamil_url"]
    }

    # Inspect each language PDF
   
    for language, pdf_url in languages.items():

        print( "\nLanguage:", language  )
          
        print( "URL:", pdf_url )

        try:

            # Download PDF
            response = requests.get( pdf_url,  timeout=30 )
                               
            response.raise_for_status()

            # Open PDF
            document = pymupdf.open(stream=response.content,filetype="pdf"  )

            text = ""

            # Extract text from every page
            for page in document:

                text += page.get_text( "text" )
                 
            text = text.strip()

            print("Pages:",  len(document) )
                
            print( "Direct characters:", len(text) )
               
            print("Sample:" )     
           
            print(text[:300] )
                
            document.close()

        except Exception as error:

            print("ERROR:", error   )
                
               
         