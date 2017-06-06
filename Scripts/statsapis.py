#%%

%reset -f

#%%

import requests
import io
import json
import pandas as pd
import sqlalchemy as sa

baseUrl = 'http://statsapi.mlb.com/api/v1/game/{!s}/feed/color'
sqlFlavor = 'sqlite'
tblName = 'raw'
dbName = 'statsapiv1.db'

engine = sa.create_engine(sqlFlavor + ':///' + dbName)

game_pk = 487637

r = requests.get(baseUrl.format(game_pk))
data = json.load(io.StringIO(r.text))

# %%

import requests


baseUrl = 'http://statsapi.mlb.com/api/v1/game/{!s}/feed/color'
game_pk = 487637

r = requests.get(baseUrl.format(game_pk))
data = r.json()
items = data['items']


#%%

try:
    items = data['items']
except KeyError as e:
    print('Not a valid game_pk')
    raise e

items.reverse()
game = pd.DataFrame()

for i in items:
    if i['group'] == 'playByPlay':
        i.update(i['data'])
        del i['data']
        game = game.append(i, ignore_index=True)

#%%

%reset -f

import requests
import json
import io

#url = 'http://gd2.mlb.com/components/game/mlb/year_2008/month_09/day_03/gid_2008_09_03_anamlb_detmlb_1/game_events.json'
url = 'http://gd2.mlb.com/components/game/mlb/year_2016/month_09/day_03/gid_2016_09_03_anamlb_seamlb_1/game_events.json'
#url = 'http://gd2.mlb.com/components/game/mlb/year_2008/month_09/day_03/gid_2008_09_03_anamlb_detmlb_1/plays.json'
#url = 'http://gd2.mlb.com/components/game/mlb/year_2016/month_09/day_03/gid_2016_09_03_anamlb_seamlb_1/plays.json'

r = requests.get(url)
data = json.load(io.StringIO(r.text))

for (key,val) in data['data']['game'].items():
    print('Key: {}, Value: {}'.format(key,val))
    print('------------------------')

#%%

%reset -f

import requests
import io
import xml.etree.ElementTree as ET

#url = 'http://gd2.mlb.com/components/game/mlb/year_2007/month_09/day_03/gid_2007_09_03_oakmlb_anamlb_1/plays.xml'
#url = 'http://gd2.mlb.com/components/game/mlb/year_2016/month_03/day_31/gid_2016_03_31_oakmlb_sfnmlb_1/plays.xml'
#url = 'http://gd2.mlb.com/components/game/mlb/year_2008/month_09/day_03/gid_2008_09_03_anamlb_detmlb_1/eventLog.xml'
#url = 'http://gd2.mlb.com/components/game/mlb/year_2008/month_09/day_03/gid_2008_09_03_anamlb_detmlb_1/game_events.xml'
#url = 'http://gd2.mlb.com/components/game/mlb/year_2016/month_09/day_03/gid_2016_09_03_anamlb_seamlb_1/game_events.xml'
#url = 'http://gd2.mlb.com/components/game/mlb/year_2008/month_02/day_26/master_scoreboard.xml'
#url = 'http://gd2.mlb.com/components/game/mlb/year_2008/month_02/day_26/gid_2008_02_29_balmlb_flomlb_1/game_events.xml'
#url = 'http://gd2.mlb.com/components/game/mlb/year_2008/month_02/day_29/master_scoreboard.xml'
#url = 'http://gd2.mlb.com/components/game/mlb/year_2008/month_02/day_28/gid_2008_02_28_bocbbc_bosmlb_1/game_events.xml'
#url = 'http://gd2.mlb.com/components/game/mlb/year_2016/month_09/day_03/master_scoreboard.xml'
url = 'http://gd2.mlb.com/components/game/mlb/year_2008/month_03/day_01/master_scoreboard.xml'

r = requests.get(url)
tree = ET.parse(io.StringIO(r.text))
root = tree.getroot()

def printxml(root, space='', recurse=True, printAttrs=True):
    print(space + 'Tag: {}'.format(root.tag))
    if printAttrs:
        print(space + 'Attributes: {}'.format(root.attrib))
    if len(root) > 0 and recurse:
        print(space + 'Children:')
        if recurse is not True:
            recurse -= 1
        for child in root:
            printxml(child, space + '    ', recurse, printAttrs)
    print(space + '--------------')

printxml(root)

#%%

%reset -f

import requests
import io

url = 'http://gd2.mlb.com/components/game/mlb/year_2008/month_09/day_03/'

r = requests.get(url)

import html

class MyHTMLParser(html.parser.HTMLParser):
    
    links = []

    def handle_starttag(self, tag, attrs):
        # Only parse the 'anchor' tag.
        if tag == "a":
           # Check the list of defined attributes.
           for name, value in attrs:
               # If href is defined, print it.
               if name == "href":
                   print(name + ': ' + value)

parser = MyHTMLParser()
parser.feed(r.text)