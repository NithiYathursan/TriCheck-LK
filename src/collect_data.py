import requests #send HTTP requests and download content from web page
from bs4 import BeautifulSoup #Parse HTML content & esy to find elements such as links and tables
from urllib.parse import urljoin,urlparse,parse_qs
import csv

#offcial Srilankan government circular page
url= "https://pubad.gov.lk/web/index.php?lang=en&option=com_circular&view=circulars"

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


# used to combine a base website URL with realtive url for creating complete url



circular_links=[]

#loop through the first 10 links found on the page
for link in links:

    #href attribute usually contains the destination url
    href=link.get("href")

    #skip links that donot have href
    if href is None:
        continue

    #Get the visible text of the link
    text=link.get_text(strip=True)

    #Convert relative url to complete url
    full_url=urljoin(url,href) 


    #Circular detail page contains "view=circular" and a circular ID "cid="
    if "view=circular" in full_url and "cid=" in full_url:

        #Add only if the URL has not already been collected
        if full_url not in circular_links:
            circular_links.append(full_url)

#shoe the no of circular detail pages were found
print("Circular links found:",len(circular_links))

#print few links
for circular_url in circular_links[:5]:
    print(circular_url)   


print("*"*40)
# inspecting the structure of each circular page

#take first circular detail page for testing
# test_circular_url=circular_links[0]

#send a request to the circular detail page
# circular_response=requests.get(test_circular_url)

#Parse the HTML of the circular detail page
# circular_soup=BeautifulSoup(circular_response.text,"html.parser")

#Find all links inside this circular detail page
# detail_links=circular_soup.find_all("a")

#Variables to store the 3 language document urls
#None - The URL has not been found yet
# english_url=None
# sinhala_url=None
# tamil_url=None




   
   
#Display the 3 document URLs.   
# print("English:",english_url)
# print("Sinhala:",sinhala_url)
# print("Tamil:",tamil_url)



#List to store the information collected from every circular
all_circulars=[]

for circular_url in circular_links:

    #open the circular detail page
    circular_response=requests.get(circular_url)

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

#count only circulars that contain all 3 languages
complete_count=0

for circular in all_circulars:
    if circular["complete_trilingual"]:
        complete_count+=1

print("Total circulars:",len(all_circulars))
print("Complete trilingual:",complete_count)
print("Incomplete:",len(all_circulars)-complete_count)        
#Display the first 3 structured records
for circular in all_circulars[:3]:

    print(circular)

#Save collected circular metadata into a CSV file
output_file="data/metadata/public_admin_circulars.csv"

with open(output_file,"w",newline="",encoding="utf-8-sig")as file:
    #Use dictonary keys as csv column names
    fieldnames=all_circulars[0].keys()
    writer=csv.DictWriter(file,fieldnames=fieldnames)

    #Write column headers
    writer.writeheader()

    #Write all circular records
    writer.writerows(all_circulars)

print(f"CSV saved as{output_file}")    