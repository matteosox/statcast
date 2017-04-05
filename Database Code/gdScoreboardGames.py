from database import database
import requests
import io
import xml.etree.ElementTree as ET
import pandas as pd
import sqlalchemy as sa
import datetime as dt

_string = sa.types.String
_integer = sa.types.Integer
_float = sa.types.Float
_date = sa.types.Date
_binary = sa.types.Binary

_baseURL = \
    'http://gd2.mlb.com/components/game/mlb/year_{yyyy}/month_{mm}/day_{dd}/{}'

dailyScoreboard = 'master_scoreboard.xml'


class db(database):
    '''Doc String'''

    dbName = 'gdScoreboardGames'
    startDate = dt.date(2008, 1, 1)
    _itemKeyName = 'game_pk'
    _username = 'matt'
    _password = 'gratitude'
    _host = 'baseball.cxx9lqfsabek.us-west-2.rds.amazonaws.com'
    _port = 5432
    _drivername = 'postgresql'
    _tblDTypes = dict(
        ampm=_string,
        aw_lg_ampm=_string,
        away_ampm=_string,
        away_code=_string,
        away_division=_string,
        away_file_code=_string,
        away_games_back=_string,
        away_games_back_wildcard=_string,
        away_league_id=_integer,
        away_league_id_spring=_integer,
        away_loss=_integer,
        away_name_abbrev=_string,
        away_split_squad=_string,
        away_sport_code=_string,
        away_team_city=_string,
        away_team_id=_integer,
        away_team_name=_string,
        away_time=_string,
        away_time_zone=_string,
        away_win=_integer,
        day=_string,
        description=_string,
        double_header_sw=_string,
        first_pitch_et=_string,
        game_data_directory=_string,
        game_nbr=_integer,
        game_pk=_integer,
        game_type=_string,
        gameday=_string,
        gameday_sw=_string,
        hm_lg_ampm=_string,
        home_ampm=_string,
        home_code=_string,
        home_division=_string,
        home_file_code=_string,
        home_games_back=_string,
        home_games_back_wildcard=_string,
        home_league_id=_integer,
        home_league_id_spring=_integer,
        home_loss=_integer,
        home_name_abbrev=_string,
        home_split_squad=_string,
        home_sport_code=_string,
        home_team_city=_string,
        home_team_id=_integer,
        home_team_name=_string,
        home_time=_string,
        home_time_zone=_string,
        home_win=_integer,
        id=_string,
        if_necessary=_string,
        league=_string,
        location=_string,
        original_date=_string,
        resume_ampm=_string,
        resume_away_ampm=_string,
        resume_away_time=_string,
        resume_date=_string,
        resume_home_ampm=_string,
        resume_home_time=_string,
        resume_time=_string,
        resume_time_date=_string,
        resume_time_date_aw_lg=_string,
        resume_time_date_hm_lg=_string,
        scheduled_innings=_integer,
        ser_games=_integer,
        ser_home_nbr=_integer,
        series=_string,
        series_num=_string,
        tbd_flag=_string,
        tiebreaker_sw=_string,
        time=_string,
        time_aw_lg=_string,
        time_date=_string,
        time_date_aw_lg=_string,
        time_date_hm_lg=_string,
        time_hm_lg=_string,
        time_zone=_string,
        time_zone_aw_lg=_integer,
        time_zone_hm_lg=_integer,
        tz_aw_lg_gen=_string,
        tz_hm_lg_gen=_string,
        venue=_string,
        venue_id=_integer,
        venue_w_chan_loc=_string,
        b=_string,
        ind=_string,
        inning=_integer,
        inning_state=_string,
        is_no_hitter=_string,
        is_perfect_game=_string,
        note=_string,
        o=_integer,
        reason=_string,
        s=_integer,
        status=_string,
        top_inning=_string)

    def _getItems(self, d):
        '''Doc string'''

        items = []
        itemKeys = []

        r = requests.get(_baseURL.format(dailyScoreboard,
                                         yyyy=d.strftime('%Y'),
                                         mm=d.strftime('%m'),
                                         dd=d.strftime('%d')))
        if r.status_code != 200:
            return (items, itemKeys)

        tree = ET.parse(io.StringIO(r.text))
        root = tree.getroot()
        games = root.findall('game')
        rowDict1 = dict.fromkeys(self._tblDTypes.keys())

        for game in games:
            itemKey = int(game.attrib['game_pk'])
            rowDict2 = rowDict1.copy()
            rowDict2.update(game.attrib)
            status = game.find('status')
            if status:
                rowDict2.update(status.attrib)
            df = pd.DataFrame(rowDict2, index=(0,))

            itemKeys.append(itemKey)
            items.append(df)

        return (items, itemKeys)
