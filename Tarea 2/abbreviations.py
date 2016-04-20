import json
import os.path
import re

from lxml import etree
from urllib.request import urlopen

def download_abbreviations():
	abbreviations = {}
	response = urlopen('http://public.oed.com/how-to-use-the-oed/abbreviations/')
	broken_html = response.read()
	html = etree.HTML(broken_html)
	tables = html.find(".//div/..[@class='page-content']").findall('.//table')
	for table in tables:
		rows = table.find('.//tbody').findall('.//tr')
		for row in rows:
			cells = row.findall('.//td')
			if len(cells)==2 and "." in cells[0].text:
				words = cells[0].text.split()
				if len(words)==1 and words[0][-1]=="." and len(words[0])>2:
					abbreviation = words[0].lower()
					meaning = cells[1].text.split(",")[0]
					meaning = re.sub(r'\([^)]*\)', '', meaning).lower()
					
					if abbreviation not in abbreviations:
						abbreviations[abbreviation] = meaning

	with open('data/abbreviations_english.json', 'w') as outfile:
		json.dump(abbreviations, outfile)

	return abbreviations

def get_abbreviations():
	file_path = 'data/abbreviations_english.json'
	if not os.path.exists(file_path):
		abbreviations = download_abbreviations()
	else:
		with open(file_path, 'r') as outfile:
			abbreviations = json.load(outfile)
	return abbreviations

abbreviations = get_abbreviations()