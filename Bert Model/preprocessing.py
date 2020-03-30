# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:50:14 2020

@author: sidhant
"""

import contractions
from bs4 import BeautifulSoup
import unicodedata
import re

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_url_special_character(text):
    for indx in range(len(text)):
#    print(processed_data.text[indx])
        text[indx]=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text[indx])
        for k in text[indx]:
            if not re.match(r"[a-zA-Z0-9@]",k):
                if k!=" ":
                    text[indx]=text[indx].replace(k,"")
    return text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text):
    return contractions.fix(text)

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

def pre_process_document(document):
    document=remove_url_special_character(document)
    # strip HTML
#    document = strip_html_tags(document)
    # lower case
    document = document.lower()
    # remove extra newlines (often might be present in really noisy text)
    document = document.translate(document.maketrans("\n\t\r", "   "))
    # remove accented characters
    document = remove_accented_chars(document)
    # expand contractions    
    document = expand_contractions(document)  
    # remove special characters and\or digits    
    # insert spaces between special characters to isolate them    
#    special_char_pattern = re.compile(r'([{.(-)!}])')
#    document = special_char_pattern.sub(" \\1 ", document)
#    document = remove_special_characters(document, remove_digits=True)  
    # remove extra whitespace
    document = re.sub(' +', ' ', document)
    document = document.strip()
    
    return document


strn="New post: Global IOT Smoke Detectors Market Revenue Strategy 2019: Credencys, Huawei, Singtel, San Jiang etc. â€“ M http://iot.ng/index.php/2019/07/12/global-iot-smoke-detectors-market-revenue-strategy-2019-credencys-huawei-singtel-san-jiang-etc-market-research-updates/Â â€¦"



out=pre_process_document(strn)
print(out)

