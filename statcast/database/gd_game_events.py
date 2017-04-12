import datetime as dt

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import sqlalchemy as sa

from .gddb import GdDatabase


_string = sa.types.String
_integer = sa.types.Integer
_float = sa.types.Float
_date = sa.types.Date
_binary = sa.types.Binary


class DB(GdDatabase):
    '''Doc String'''

    dbName = 'gdGameEvents'
    startDate = dt.date(2008, 1, 1)
    _fileName = 'game_events.xml'
    _tblDTypes = dict(
        game_pk=_integer,
        inning=_integer,
        inning_topbot=_string,
        entry=_string,
        away_team_runs=_integer,
        b=_integer,
        b1=_integer,
        b2=_integer,
        b3=_integer,
        batter=_integer,
        des=_string,
        des_es=_string,
        events=_string,
        events_es=_string,
        event_num=_integer,
        home_team_runs=_integer,
        num=_integer,
        o=_integer,
        pitcher=_integer,
        play_guid=_string,
        pitch=_integer,
        player=_integer,
        rbi=_integer,
        s=_integer,
        score=_string,
        tfs=_integer,
        tfs_zulu=_string,
        pitch_des=_string,
        pitch_des_es=_string,
        pitch_type=_string,
        start_speed=_float,
        sv_id=_string,
        type=_string)

    def _parseFile(self, file, itemKey):
        '''Doc string'''

        rowDict1 = dict.fromkeys(self._tblDTypes.keys(), np.nan)

        tree = ET.parse(file)
        root = tree.getroot()
        innings = root.findall('inning')
        rowDict1[self._itemKeyName] = itemKey
        df = pd.DataFrame()
        for inning in innings:
            rowDict1['inning'] = inning.get('num')
            for innHalf in inning:
                if innHalf.tag == 'top':
                    rowDict1['inning_topbot'] = 'top'
                else:
                    rowDict1['inning_topbot'] = 'bot'
                for entry in innHalf:
                    rowDict2 = rowDict1.copy()
                    rowDict2['entry'] = entry.tag
                    rowDict2['tfs'] = entry.attrib.pop('start_tfs', np.nan)
                    rowDict2['tfs_zulu'] = \
                        entry.attrib.pop('start_tfs_zulu', np.nan)
                    rowDict2['events'] = \
                        '::'.join(entry.attrib.pop(key)
                                  for key in sorted(tuple(entry.attrib.keys()))
                                  if 'event' in key and
                                  not key.endswith(('_es', '_num')))
                    rowDict2['events_es'] = \
                        '::'.join(entry.attrib.pop(key)
                                  for key in sorted(tuple(entry.attrib.keys()))
                                  if 'event' in key and
                                  key.endswith('_es'))
                    rowDict2.update(entry.attrib)
                    pitches = entry.findall('pitch')
                    if entry.tag == 'atbat' and len(pitches) > 0:
                        for pitch in pitches:
                            rowDict3 = rowDict2.copy()
                            rowDict3['pitch_des'] = \
                                pitch.attrib.pop('des', np.nan)
                            rowDict3['pitch_des_es'] = \
                                pitch.attrib.pop('des_es', np.nan)
                            rowDict3.update(pitch.attrib)
                            df = df.append(pd.DataFrame(rowDict3, index=(0,)),
                                           ignore_index=True)
                    else:
                        df = df.append(pd.DataFrame(rowDict2, index=(0,)),
                                       ignore_index=True)

        df.replace('', np.nan, inplace=True)
        return df

    def _fixItem(self, item, bads, itemKey):
        '''Doc String'''

        for ind, col, err in bads:
            if col in ('b1', 'b2', 'b3'):
                badElem = item.loc[ind, col]
                goodElem = item.loc[ind, col].split(' ')[-1]
                item.loc[ind, col] = goodElem
                self.logger.info(
                    'Bad element {} at row {}, column {} of item {} was '
                    'replaced with {}.'.format(badElem, ind, col,
                                               itemKey, goodElem))
            else:
                super()._fixItem(item, bads, itemKey)
