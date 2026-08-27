import csv

file_path = "data/metadata/public_admin_circulars.csv"

with open(file_path, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

print("Saved records:", len(rows))