import csv

#Path of the metadata collected by te scraper
csv_file="data/metadata/public_admin_circulars.csv"

#Read all circular records from the csv
with open(csv_file,"r",encoding="utf-8-sig") as file:
    reader=csv.DictReader(file)
    rows=list(reader)

print("Total records:",len(rows))


#Collect all circular IDs
circular_ids=[]

for row in rows:
    circular_ids.append(row["circular_id"])

#Compare total IDs with unique IDs
unique_ids=set(circular_ids)

print("Unique circular IDs:",len(unique_ids))
print("Duplicate IDs:",len(circular_ids)-len(unique_ids))

#Meta data fields to validate
fields_to_check =[
    "circular_number",
    "circular_name",
    "year",
    "circular_date",
    "english_url",
    "sinhala_url",
    "tamil_url"
]

print("\n Missing Values:")

for field in fields_to_check:
    missing_count=0
    for row in rows:
        #csv missing values may appear as an empty string
        if row[field].strip()=="":
            missing_count+=1

    print(f"{field}: {missing_count}")        


#Verify that complete_trilingual agrees with the three language URL columns
mismatch_count=0

for row in rows:
    has_all_three=(
        row["english_url"].strip()!="" and row["sinhala_url"].strip()!="" and row["tamil_url"].strip()!=""

    )
    stored_flag=row["complete_trilingual"].strip().lower()=="true"

    if has_all_three != stored_flag:
        mismatch_count+=1

print("\n Trilingual flag mismatches:",mismatch_count)        

#count high-quality records suited for NLP analysis
usable_records=[]

for row in rows:
    has_all_languages=(row["complete_trilingual"].strip().lower()=="true")
    has_required_metadata=(row["circular_number"].strip()!="" and row["circular_name"].strip()!="")

    if has_all_languages and has_required_metadata:
        usable_records.append(row)

print("\nUsable NLP candidate records:",len(usable_records))        

#Path to save records suitable forNLP processing
candidate_file="data/metadata/public_admin_nlp_candidates.csv"

with open(candidate_file,"w",newline="",encoding="utf-8-sig")as file:

    #Use the same columns as the original dataset
    fieldnames=usable_records[0].keys()

    writer=csv.DictWriter(file,fieldnames=fieldnames)

    writer.writeheader()

    writer.writerows(usable_records)

print("NLP candidate dataset saved:",candidate_file)    