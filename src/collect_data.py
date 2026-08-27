import requests #send HTTP requests and download content from web page
from bs4 import BeautifulSoup #Parse HTML content & esy to find elements such as links and tables
from urllib.parse import urljoin,urlparse,parse_qs  ## used to combine a base website URL with realtive url for creating complete url
import csv
import time 

#offcial Srilankan government circular page
url = "https://pubad.gov.lk/web/index.php?lang=en&option=com_circular&view=circulars"

#Send a GET request to the website
response=requests.get(url)

#check whether the request was successful
print("Status Code:",response.status_code)

#Convert the raw HTML into BeautifulSoup Object,makes it easier to search the web page structure

soup=BeautifulSoup(response.text,"html.parser")

# print the page title to confirm that we loaded the correct page
print("Page Title:",soup.title)

#Find all <a> tags because hyperlinks in HTML
links=soup.find_all("a")

#print the count of links on that web page
print("Total links found:",len(links))


#Store circular detail links collected from multiple pages
circular_links=[]

#Store previous collected URLs so duplicates are avoided
seen_links = set()

# Start from the first listing page
offset = 0

while True:

    # Create the current listing page URL
    page_url = url + f"&limitstart={offset}"

    print("Reading listing page:", page_url)

    page_response = requests.get(
    page_url,
    timeout=30
    )

    # Stop the program if the website returns
    # an unsuccessful HTTP status code.
    page_response.raise_for_status()

    page_soup = BeautifulSoup(
        page_response.text,
        "html.parser"
    )

    page_links = page_soup.find_all("a")

    # Count how many NEW circular links
    # are found on this particular page.
    new_links_count = 0

    for link in page_links:

        href = link.get("href")

        if href is None:
            continue

        full_url = urljoin(page_url, href)

        if (
            "view=circular" in full_url
            and "cid=" in full_url
        ):

            if full_url not in seen_links:

                seen_links.add(full_url)
                circular_links.append(full_url)

                new_links_count += 1

    print(
        "New circulars on this page:",
        new_links_count
    )

    # If the page gives no new circulars,
    # we have reached the end of the archive.
    if new_links_count == 0:
        break
    
    # Move to the next listing page.
    # Each page contains 10 circulars.
    offset += 10

    # Small delay before requesting next page.
    time.sleep(0.5)


print(
    "Total circular links found:",
    len(circular_links)
)



print("*" * 40)

#Save collected circular metadata into a CSV file
output_file="data/metadata/public_admin_circulars.csv"

#List to store the information collected from every circular
all_circulars=[]

# Store URLs that could not be processed
failed_circulars = []

for index, circular_url in enumerate(circular_links, start=1):

    try:
        # Open the circular detail page
        circular_response = requests.get(
            circular_url,
            timeout=30
        )

        circular_response.raise_for_status()

    except requests.RequestException as error:

        print(
            f"Failed {index}/{len(circular_links)}:",
            circular_url
        )

        print("Reason:", error)

        failed_circulars.append(circular_url)

        # Skip this circular and continue with the next one
        continue
    #convert the HTML into a BeautifulSoup object
    circular_soup=BeautifulSoup(circular_response.text,"html.parser")

    #Find all links available inside this circular page
    detail_links=circular_soup.find_all("a")

    #Find the main metadata table
    metadata_table=circular_soup.find("table")

    #Variables are initially None in case a field 
    circular_name=None
    circular_number=None
    branch_name=None
    year=None
    circular_date=None

    if metadata_table is not None:
        #Read every row in the table
        for row in metadata_table.find_all("tr"):
            #row may contain multiple label-value pairs
            cells=row.find_all(["th","td"])

            #process cells two at a time label,value
            for i in range(0,len(cells)-1,2):
                label=cells[i].get_text("",strip=True)
                value=cells[i+1].get_text("",strip=True)

                if label=="Circular Name":
                    circular_name=value
                elif label=="Circular Number":
                    circular_number=value
                elif label=="Branch Name":
                    branch_name=value
                elif label=="Year":
                    year=value
                elif label=="Circular Date":
                    circular_date=value

                                    
    #Initially we set to None because no 3 ;anguage documents
    english_url=None
    sinhala_url=None
    tamil_url=None    

    for link in detail_links:

        #Get the href value
        href=link.get("href")

        #skip links without href
        if href is None:
            continue

        #Get the visible text of the link    
        text=link.get_text(strip=True)

        #Convert relative URL into a complete URL
        full_url=urljoin(circular_url,href)

        #Identify the actual document language using visible text shown on the website
        if text=="In English":
            english_url=full_url
        elif text=="In Sinhala":
            sinhala_url=full_url
        elif text=="In Tamil":
            tamil_url=full_url

    parsed_url=urlparse(circular_url)
    query_parameters=parse_qs(parsed_url.query)
    circular_id=query_parameters.get("cid",[None])[0]

    #check whether English,Tamil,Sinhala documents are all available
    is_complete=(
        english_url is not None and sinhala_url is not None and tamil_url is not None)


    #Store all important information about one circular inside a Python Dictionary
    circular_data={
        "circular_id":circular_id,
        "circular_number":circular_number,
        "circular_name":circular_name,
        "branch_name":branch_name,
        "year":year,
        "circular_date":circular_date,
        "circular_url":circular_url,
        "english_url":english_url,
        "sinhala_url":sinhala_url,
        "tamil_url":tamil_url,
        "complete_trilingual":is_complete
    }

    all_circulars.append(circular_data)
    # Show current progress
    print(f"Processed {index}/{len(circular_links)}")

    # Save collected records after every 50 processed pages
    if index % 50 == 0:

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            fieldnames = all_circulars[0].keys()

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(all_circulars)

        print(
            f"Checkpoint saved: {len(all_circulars)} records"
        )
    time.sleep(0.3)

#count only circulars that contain all 3 languages
complete_count=0

for circular in all_circulars:
    if circular["complete_trilingual"]:
        complete_count+=1

print("Total circulars:",len(all_circulars))
print("Complete trilingual:",complete_count)
print("Incomplete:",len(all_circulars)-complete_count)  
print("Failed circulars:", len(failed_circulars))      

#Display the first 3 structured records
for circular in all_circulars[:3]:

    print(circular)



with open(output_file,"w",newline="",encoding="utf-8-sig")as file:
    #Use dictonary keys as csv column names
    fieldnames=all_circulars[0].keys()
    writer=csv.DictWriter(file,fieldnames=fieldnames)

    #Write column headers
    writer.writeheader()

    #Write all circular records
    writer.writerows(all_circulars)

print(f"CSV saved as : {output_file}")    

