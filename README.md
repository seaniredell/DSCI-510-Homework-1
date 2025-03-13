# DSCI-510-Homework-1
Homework 1

Code Specifications that must be met to be able to run the scrips dealing with Tikka Similarity and Etllib 

Specifications:
- Rename any columns that have "date" in the name
- Pynev must be installed and python should be updated to 2.7.18 (Etllib will not run on more modern versions)
- csvkit must be installed
- homebrew must be installed
- libmagic should be installed using homebrew
- ETLLib should be installed
- hirelite must be installed (IMPORTANT! must be installed on both python 2.7.18 and 3.1 (or whatever python 3.X you have installed)
- iso8601 must be installed

Data Extraction Considerations:
- During data extraction using requests and BeautifulSoup to scrape datasets from World Population Review, I encountered issues with webpage structure changes. For example, a newly added table on one of the websites initially caused my original code to fail. I revised and tested the script to ensure that it was functional at the time of submission (March 14). However, if the webpage layout changes in the future, additional modifications to the scraping script may be necessary.









Sean's Contribution: Part 6 and 7 Tika Similarity and roughly 1/2 the report that deals with findings

Yingyi's Contribution: 
- Added 4 new features to the Haunted Places dataset: Audio Evidence, Image/Video/Visual Evidence, Haunted Places Witness Count, and Apparition Type.
- Scraped, cleaned, and merged datasets with MIME Type Message/HTTP, extracting 3 features for further analysis.
