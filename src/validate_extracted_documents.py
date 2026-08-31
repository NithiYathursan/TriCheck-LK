import json
from collections import Counter


input_file = (
    "data/processed/extracted_documents_full.jsonl"
)

records = []

# Read extracted JSONL file

with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        records.append(record)

print("Total extracted records:", len(records))  

# Count languages

language_counts = Counter(
    record["language"]
    for record in records
)

print(  "\nLanguage counts:",language_counts)
  
# Count extraction statuses

status_counts = Counter(
    record["extraction_status"]
    for record in records
)

print("\nExtraction status counts:", status_counts)
    
# Check empty text

empty_records = [
    record
    for record in records
    if len(record["text"].strip()) == 0
]

print( "\nEmpty text records:",  len(empty_records))
  
# Show circulars and metadata years
# Use circular_id because circular_number may repeat


print("\nCirculars:")

seen_ids = set()

for record in records:

    circular_id = record["circular_id"]

    if circular_id not in seen_ids:

        print(
            record["circular_number"],
            "→ metadata year:",
            record["year"]
        )

        seen_ids.add(circular_id)

# Show OCR extracted records

ocr_records = [
    record
    for record in records
    if record["extraction_status"] == "OCR_TEXT"
]

print( "\nOCR extracted records:", len(ocr_records))
   
for record in ocr_records:

    print(
        record["circular_number"],
        "→",
        record["language"],
        "→",
        record["character_count"],
        "characters"
    )

# Show failed / unusable records

failed_records = [
    record
    for record in records
    if record["extraction_status"]
    in ["ERROR", "UNUSABLE"]
]

print( "\nFailed records:", len(failed_records))

for record in failed_records:

    print(
        record["circular_number"],
        "→",
        record["language"],
        "→",
        record["extraction_status"]
    )

# Group translations by unique circular ID

records_by_circular = {}

for record in records:

    circular_id = record["circular_id"]

    if circular_id not in records_by_circular:
        records_by_circular[circular_id] = []

    records_by_circular[circular_id].append( record )
    
# Check large character-count differences

print(
    "\nPotential extraction quality issues:"
)

quality_issue_count = 0


for circular_id, circular_records in records_by_circular.items():

    circular_number = circular_records[0][ "circular_number" ]
       
    # Only compare complete 3-language groups
    if len(circular_records) != 3:
        continue


    character_counts = [
        record["character_count"]
        for record in circular_records
        if record["character_count"] > 0
    ]


    if len(character_counts) != 3:
        continue


    smallest_count = min( character_counts    )
       
    largest_count = max(  character_counts )
      
    # Flag when one translation is more than
    # 5 times longer than another
    ratio = ( largest_count / smallest_count )
       
    if ratio > 5:

        quality_issue_count += 1

        print(
            f"\n{circular_number}"
        )

        for record in circular_records:

            print(
                " ",
                record["language"],
                "→",
                record["character_count"],
                "→",
                record["extraction_status"]
            )


        print(  "Length ratio:",  round(ratio, 2))
          
print( "\nPotential quality issues:", quality_issue_count)
   
# Check repeated circular numbers

number_counts = Counter(
    record["circular_number"]
    for record in records
    if record["language"] == "English"
)

print("\nRepeated circular numbers:")
    
repeated_count = 0

for number, count in number_counts.items():

    if count > 1:

        repeated_count += 1

        print( number, "→",  count,  "records"  )
           
print( "Total repeated circular numbers:", repeated_count)

# Check actual duplicate circular IDs

english_records = [
    record
    for record in records
    if record["language"] == "English"
]

id_counts = Counter(
    record["circular_id"]
    for record in english_records
)

duplicate_ids = {
    circular_id: count
    for circular_id, count in id_counts.items()
    if count > 1
}

print(
    "\nDuplicate circular IDs:",
    len(duplicate_ids)
)

for circular_id, count in duplicate_ids.items():
    print(
        circular_id,
        "→",
        count,
        "records"
    )

# Check every circular has exactly 3 languages

incomplete_groups = []

for circular_id, circular_records in records_by_circular.items():

    languages = {
        record["language"]
        for record in circular_records
    }

    if (
        len(circular_records) != 3
        or languages != {
            "English",
            "Sinhala",
            "Tamil"
        }
    ):
        incomplete_groups.append(
            circular_id
        )

print(
    "\nIncomplete trilingual groups:",
    len(incomplete_groups)
)


# Final usable dataset summary


usable_records = [
    record
    for record in records
    if record["extraction_status"]
    in ["DIRECT_TEXT", "OCR_TEXT"]
    and len(record["text"].strip()) > 0
]

print(
    "\nUsable document records:",
    len(usable_records)
)


# Check completely usable trilingual circulars
usable_by_circular = {}

for record in usable_records:

    circular_id = record["circular_id"]

    if circular_id not in usable_by_circular:
        usable_by_circular[circular_id] = []

    usable_by_circular[circular_id].append(record)


complete_usable_circulars = 0

for circular_id, circular_records in usable_by_circular.items():

    languages = {
        record["language"]
        for record in circular_records
    }

    if languages == {
        "English",
        "Sinhala",
        "Tamil"
    }:
        complete_usable_circulars += 1


print(
    "Complete usable trilingual circulars:",
    complete_usable_circulars
)

print(
    "Complete usable documents:",
    complete_usable_circulars * 3
)   

# Save only complete usable trilingual groups

complete_usable_ids = set()

for circular_id, circular_records in usable_by_circular.items():

    languages = {
        record["language"]
        for record in circular_records
    }

    if languages == {
        "English",
        "Sinhala",
        "Tamil"
    }:
        complete_usable_ids.add(
            circular_id
        )


final_records = [
    record
    for record in usable_records
    if record["circular_id"]
    in complete_usable_ids
]


final_output_file = (
    "data/processed/"
    "extracted_documents_clean.jsonl"
)


with open(
    final_output_file,
    "w",
    encoding="utf-8"
) as file:

    for record in final_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


print(
    "\nFinal clean records:",
    len(final_records)
)

print(
    "Final clean circulars:",
    len(complete_usable_ids)
)

print(
    "Saved to:",
    final_output_file
)