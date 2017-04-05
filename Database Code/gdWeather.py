import pandas as pd
import sqlalchemy as sa
import datetime as dt
from gdDatabase import gdDatabase
import xml.etree.ElementTree as ET

_string = sa.types.String
_integer = sa.types.Integer
_float = sa.types.Float
_date = sa.types.Date


class db(gdDatabase):
    '''Doc String'''

    dbName = 'gdWeather'
    startDate = dt.date(2008, 1, 1)
    _fileName = 'plays.xml'
    _tblDTypes = dict(
        condition=_string,
        temp=_integer,
        wind=_string,
        game_pk=_integer)

    def _parseFile(self, file, itemKey):
        '''Doc string'''

        tree = ET.parse(file)
        root = tree.getroot()
        weather = root.find('weather').attrib

        df = pd.DataFrame({'condition': weather['condition'],
                           'temp': int(weather['temp']),
                           'wind': weather['wind'],
                           self._itemKeyName: itemKey}, index=(0,))

        return df
