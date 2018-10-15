from bs4 import BeautifulSoup as bs
import urllib2
import regex as re
import time


class setPair:

    def __init__(self):
        self.pre_reqs = set([])
        self.quarters_offered = [0]*4

    #not yet implemented; ignores that some classes may be taken concurrently as of now
    def add_prereq(self, pre_req):
        self.pre_reqs.add(pre_req)

    # works
    def add_quarter(self, qrt):
        pat = re.compile(r'(A)?(W)?(Sp)?(S)?')
        matches = pat.finditer(qrt)
        for match in matches:
            for i in range(1,5):
                if not match.group(i) == None:
                    self.quarters_offered[i-1] += 1

    def to_string(self):
        return "prereqs: " + str(self.pre_reqs) + ", quarters offered: " + str(self.quarters_offered)




quote_page = 'https://www.washington.edu/students/crscat/appmath.html'
page = urllib2.urlopen(quote_page)

soup = bs(page, 'html.parser')

class_code = re.compile(r'[a-zA-Z\s]{3,5}\d{3}')
matches = class_code.findall(str(soup))

classesAvailable = dict()

for class_info in soup.find_all("a", attrs={"name": class_code} ):
    class_num = class_info['name']
    print class_num

    credit_pattern = re.compile(r'\(.{1,10}\)')
    credit_count = "".join(credit_pattern.findall(str(class_info)))[1:-1]
    print credit_count

    prereq_pattern = re.compile(r'(Prerequisite:.*?)(\.\s)')
    prereqs = prereq_pattern.findall(str(class_info))
    print prereqs

    qrt_pattern = re.compile(r'(Offered:.*?)\.')
    qrts_offered = "".join(qrt_pattern.findall(str(class_info))).replace("Offered: ", "")[-6:]
    print qrts_offered

    new_pair = setPair()
    new_pair.add_quarter(qrts_offered)
    print new_pair.to_string()
    classesAvailable[str(class_num)] = new_pair



