import abc
import time
import io

import xml.etree.ElementTree as ET

import requests

from .database import Database


_baseurl = \
    'http://gd2.mlb.com/components/game/mlb/year_{yyyy}/month_{mm}/day_{dd}/{}'

dailyScoreboard = 'master_scoreboard.xml'


class GdDatabase(Database, metaclass=abc.ABCMeta):
    '''Doc String'''

    _itemKeyName = 'game_pk'
    _username = 'matt'
    _password = 'gratitude'
    _host = 'baseball.cxx9lqfsabek.us-west-2.rds.amazonaws.com'
    _port = 5432
    _drivername = 'postgresql'

    @abc.abstractmethod
    def _fileName():
        pass

    @abc.abstractmethod
    def _parseFile(self, file, itemKey):
        pass

    def _getItems(self, d):
        '''Doc string'''

        items = []
        itemKeys = []
        url = _baseurl.format(dailyScoreboard,
                              yyyy=d.strftime('%Y'),
                              mm=d.strftime('%m'),
                              dd=d.strftime('%d'))

        for dummy in range(100):
            try:
                r1 = requests.get(url)
            except Exception as e:
                self.logger.debug(
                    '{!r} occurred while trying to dowload {}.'.
                    format(e, url))
                time.sleep(5)
            else:
                break
        else:
            self.logger.error(
                'Unable to download {} after {} attempts.'.
                format(url, dummy + 1))
            return (items, itemKeys)

        if r1.status_code != 200:
            return (items, itemKeys)

        tree = ET.parse(io.StringIO(r1.text))
        root = tree.getroot()
        games = root.findall('game')

        for game in games:
            itemKey = int(game.attrib['game_pk'])
            gid = game.attrib['gameday']
            url = _baseurl.format('gid_' + gid + '/' + self._fileName,
                                  yyyy=d.strftime('%Y'),
                                  mm=d.strftime('%m'),
                                  dd=d.strftime('%d'))

            for dummy in range(100):
                try:
                    r2 = requests.get(url)
                except Exception as e:
                    self.logger.debug(
                        '{!r} occurred while trying to dowload {}.'.
                        format(e, url))
                    time.sleep(5)
                else:
                    break
            else:
                self.logger.error(
                    'Unable to download {} after {} attempts.'.
                    format(url, dummy + 1))
                continue

            if r2.status_code != 200:
                try:
                    status = game.find('status').attrib['status']
                except:
                    status = None

                gidParts = gid.rsplit('_', 3)
                gameDate = gidParts[0]
                awayLg = gidParts[1][3:]
                awayTm = gidParts[1][:3]
                homeLg = gidParts[2][3:]
                homeTm = gidParts[2][:3]

                if not awayLg == homeLg == 'mlb':
                    self.logger.info(
                        '''
Received {} status code while trying to retrieve {} for gid = {},
game_pk = {} at address {}, determined game involved non-MLB team'''.
                        format(r2.status_code, self._fileName, gid, itemKey,
                               url))
                elif awayTm == homeTm:
                    self.logger.info(
                        '''
Received {} status code while trying to retrieve {} for gid = {},
game_pk = {} at address {}, determined game is intra-squad'''.
                        format(r2.status_code, self._fileName, gid, itemKey,
                               url))
                elif not d.strftime('%Y_%m_%d') == gameDate:
                    self.logger.info(
                        '''
Received {} status code while trying to retrieve {} for gid = {},
game_pk = {} at address {}, determined game occurred on different date'''.
                        format(r2.status_code, self._fileName, gid, itemKey,
                               url))
                elif status is None:
                    self.logger.warning(
                        '''
Received {} status code while trying to retrieve {} for gid = {},
game_pk = {} at address {}, could not determine game status'''.
                        format(r2.status_code, self._fileName, gid, itemKey,
                               url))
                elif not status == 'Final':
                    self.logger.info(
                        '''
Received {} status code while trying to retrieve {} for gid = {},
game_pk = {} at address {}, determined game status was {}'''.
                        format(r2.status_code, self._fileName, gid, itemKey,
                               url, status))
                else:
                    self.logger.warning(
                        '''
Received {} status code while trying to retrieve {} for gid = {},
game_pk = {} at address {}, could not determine cause'''.
                        format(r2.status_code, self._fileName, gid, itemKey,
                               url))
                continue

            itemKeys.append(itemKey)
            items.append(self._parseFile(io.StringIO(r2.text), itemKey))

        return (items, itemKeys)
