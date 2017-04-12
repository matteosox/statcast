import logging
import abc

from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd
import sqlalchemy as sa

from ..tools.fixpath import findFile


_string = sa.types.String
_integer = sa.types.Integer
_float = sa.types.Float
_date = sa.types.Date
_binary = sa.types.Binary


class Database(metaclass=abc.ABCMeta):
    '''Doc String'''

    _username = None
    _password = None
    _host = None
    _port = None
    _tblName = 'raw'
    _updtTblName = 'updates'
    _updtTblDTypes = {'cmd': _string, 'dateFrom': _date, 'dateTo': _date}

    @abc.abstractmethod
    def _drivername():
        pass

    @abc.abstractmethod
    def _tblDTypes():
        pass

    @abc.abstractmethod
    def dbName():
        pass

    @abc.abstractmethod
    def startDate():
        pass

    @abc.abstractmethod
    def _itemKeyName():
        pass

    @abc.abstractmethod
    def _getItems(self, date):
        pass

    def __init__(self, fast=False):
        '''Doc string'''

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        fmt = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        sH = logging.StreamHandler()
        sH.setLevel(logging.WARNING)
        sH.setFormatter(fmt)
        self.logger.addHandler(sH)

        # Local database
        if self._host is None:
            dbPath = findFile(self.dbName + '.db')
            if not dbPath:
                dbPath = self.dbName + '.db'
            logPath = str(Path(dbPath).with_name(self.dbName + '.log'))
        else:
            dbPath = self.dbName
            logPath = self.dbName + '.log'

        fH = logging.FileHandler(logPath)
        fH.setLevel(logging.DEBUG)
        fH.setFormatter(fmt)
        self.logger.addHandler(fH)

        url = sa.engine.url.URL(drivername=self._drivername,
                                username=self._username,
                                password=self._password,
                                host=self._host,
                                port=self._port,
                                database=dbPath)
        self.engine = sa.create_engine(url)

        if self._drivername == 'postgresql':
            tempURL = sa.engine.url.URL(drivername=self._drivername,
                                        username=self._username,
                                        password=self._password,
                                        host=self._host,
                                        port=self._port,
                                        database='postgres')
            tempEngine = sa.create_engine(tempURL)
            dbs = pd.read_sql_query(
                    '''SELECT datname FROM pg_database
                    WHERE datistemplate = false''', tempEngine)
            if not (self.dbName == dbs).any()[0]:
                tempConnection = tempEngine.connect()
                tempConnection.execute('commit')
                tempConnection.execute(
                        'create database "{}"'.format(self.dbName))
                self._init0()
                return

        if not self.engine.has_table(self._tblName):
            self._init0()
            return

        self.lastUpdate = pd.read_sql_query(
            'SELECT "dateTo" FROM "{}" ORDER BY "dateTo" DESC LIMIT 1'.
            format(self._updtTblName),
            self.engine, parse_dates=['dateTo']).dateTo.iloc[0].date()

        if fast:
            self.itemKeys = None
            return

        self.itemKeys = list(pd.read_sql_query(
            'SELECT DISTINCT "{}" FROM "{}"'.format(self._itemKeyName,
                                                self._tblName),
            self.engine)[self._itemKeyName])

        if not self.lastUpdate == dt.date.today():
            self.update()

    def _init0(self):
        '''Doc string'''

        self.logger.info('Initializing database')
        self.itemKeys = []

        self._update(self.startDate)

    def _addItem(self, item, itemKey, replace=False):
        '''Doc String'''

        if replace:
            self._rmItem(itemKey)

        try:
            item.to_sql(self._tblName, self.engine, if_exists='append',
                        index=False, dtype=self._tblDTypes)
        except Exception as e:
            bads = self._checkItem(item)
            if not bads:
                raise e
            else:
                self._fixItem(item, bads, itemKey)

            item.to_sql(self._tblName, self.engine, if_exists='append',
                        index=False, dtype=self._tblDTypes)

        self.itemKeys.append(itemKey)

    def _addDate(self, d, replace=False):
        '''Doc string'''

        (items, itemKeys) = self._getItems(d)
        for (item, itemKey) in zip(items, itemKeys):
            self._addItem(item, itemKey, replace)

    def _addDates(self, dates, replace=False):
        '''Doc string'''

        for date in dates:
            self._addDate(date, replace)

    def _addDateRng(self, start, end=dt.date.today(), step=1, replace=False):
        '''Doc string'''

        dates = [start + dt.timedelta(ii)
                 for ii in range(0, (end - start).days, step)]
        self._addDates(dates, replace)

    def _rmItem(self, itemKey):
        '''Doc String'''

        self.engine.execute('DELETE FROM "{}" WHERE "{}" = {}'.
                            format(self._tblName, self._itemKeyName, itemKey))
        if itemKey in self.itemKeys:
            self.itemKeys.remove(itemKey)

    def _rmDate(self, d):
        '''Doc String'''

        (items, itemKeys) = self._getItems(d)
        for itemKey in itemKeys:
            self._rmItem(itemKey)

    def _rmDates(self, dates):
        '''Doc String'''

        for date in dates:
            self._rmDate(date)

    def _rmDateRng(self, start, end=dt.date.today(), step=1):
        '''Doc String'''

        dates = [start + dt.timedelta(ii)
                 for ii in range(0, (end - start).days, step)]
        self._rmDates(dates)

    def _addUpdate(self, cmd, dateFrom, dateTo):
        '''Doc String'''

        if self.engine.has_table(self._updtTblName):
            count = self.engine.execute(
                    'SELECT COUNT(*) FROM "{}"'.format(self._updtTblName)). \
                fetchone()
        else:
            count = (0,)
        update = pd.DataFrame({'cmd': cmd,
                               'dateFrom': dateFrom,
                               'dateTo': dateTo},
                              index=count)
        update.to_sql('updates', self.engine, if_exists='append',
                      dtype=self._updtTblDTypes)
        self.lastUpdate = dateTo

    def _update(self, start, end=dt.date.today(), replaceStart=False):
        '''Doc String'''

        if replaceStart:
            self._rmDate(start)

        self._addDateRng(start, end)
        self._addUpdate('update', start, end)
        self.logger.info('Updated database')

    def update(self):
        '''Doc String'''

        if not self.engine.has_table(self._tblName):
            print('Database not yet initialized')
            return

        self._update(self.lastUpdate, replaceStart=True)

    def loadItem(self, itemKey):
        '''Doc String'''

        if itemKey not in self.itemKeys:
            print('Item key {} not found in database'.format(itemKey))
            return pd.DataFrame()

        return pd.read_sql_query(
            'SELECT * FROM "{}" WHERE "{}" = {}'.format(
                self._tblName, self._itemKeyName, itemKey),
            self.engine, parse_dates=[k for k, v in self._tblDTypes.items()
                                      if v == _date])

    def _checkItem(self, item):
        '''Doc String'''

        bads = []
        for col, sqlType in self._tblDTypes.items():
            ser = item.loc[:, col]
            if sqlType is _string:
                checkFunc = str
            elif sqlType is _integer:
                checkFunc = int
            elif sqlType is _float:
                checkFunc = float
            elif sqlType is _date:
                checkFunc = pd.to_datetime
            elif sqlType is _binary:
                checkFunc = bool
            else:
                raise TypeError('An invalid datatype {} was supplied for '
                                'column {}'.format(sqlType, col))
            for ind, elem in ser.iloc[~(ser.isnull().values)].items():
                try:
                    checkFunc(elem)
                except Exception as e:
                    bads.append((ind, col, e))
        return bads

    def _fixItem(self, item, bads, itemKey):
        '''Doc String'''

        for bad in bads:
            elem = item.loc[bad[0], bad[1]]
            item.loc[bad[0], bad[1]] = np.nan
            self.logger.warning(
                'Bad element {} at row {}, column {} of item {} was replaced '
                'with np.nan. This exception was raised: {!r}'.format(elem,
                                                                      bad[0],
                                                                      bad[1],
                                                                      itemKey,
                                                                      bad[2]))
