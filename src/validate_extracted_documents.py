import json
from collections import Counter


input_file = (
    "data/processed/extracted_documents.jsonl"
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

   
   