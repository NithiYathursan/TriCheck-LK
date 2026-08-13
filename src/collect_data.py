import requests #send HTTP requests and download content from web page
from bs4 import BeautifulSoup #Parse HTML content & esy to find elements such as links and tables

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