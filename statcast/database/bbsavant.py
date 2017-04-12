import time
import datetime as dt

import pandas as pd
import sqlalchemy as sa

from .database import Database


_string = sa.types.String
_integer = sa.types.Integer
_float = sa.types.Float
_date = sa.types.Date

_baseURL = '''
https://baseballsavant.mlb.com/statcast_search/csv?
all=true&
hfPT=&
hfZ=&
hfGT=R%7CPO%7CS%7C&
hfPR=&
hfAB=&
stadium={venue}&
hfBBT=&
hfBBL=&
hfC=&
season=all&
player_type=batter&
hfOuts=&
pitcher_throws=&
batter_stands=&
start_speed_gt=&
start_speed_lt=&
perceived_speed_gt=&
perceived_speed_lt=&
spin_rate_gt=&
spin_rate_lt=&
exit_velocity_gt=&
exit_velocity_lt=&
launch_angle_gt=&
launch_angle_lt=&
distance_gt=&
distance_lt=&
batted_ball_angle_gt=&
batted_ball_angle_lt=&
game_date_gt={date}&
game_date_lt={date}&
team=&
position=&
hfRO=&
home_road=&
hfInn=&
min_pitches=0&
min_results=0&
group_by=name-event&
sort_col=pitches&
player_event_sort=start_speed&
sort_order=desc&
min_abs=0&
xba_gt=&
xba_lt=&
px1=&
px2=&
pz1=&
pz2=&
ss_gt=&
ss_lt=&
is_barrel=&
type=details&
'''.replace('\n', '')

_venues = [
    'LAA',
    'HOU',
    'OAK',
    'TOR',
    'ATL',
    'MIL',
    'STL',
    'CHC',
    'ARI',
    'LAD',
    'SF',
    'CLE',
    'SEA',
    'MIA',
    'NYM',
    'WSH',
    'BAL',
    'SD',
    'PHI',
    'PIT',
    'TEX',
    'TB',
    'BOS',
    'CIN',
    'COL',
    'KC',
    'DET',
    'MIN',
    'CWS',
    'NYY']


class DB(Database):
    '''Doc String'''

    dbName = 'bbsavant'
    _username = 'matt'
    _password = 'gratitude'
    _host = 'baseball.cxx9lqfsabek.us-west-2.rds.amazonaws.com'
    _port = 5432
    _drivername = 'postgresql'
    startDate = dt.date(2008, 1, 1)
    _itemKeyName = 'game_pk'
    _tblDTypes = dict(
        pitch_type=_string,
        pitch_id=_integer,
        game_date=_date,
        start_speed=_float,
        x0=_float,
        z0=_float,
        player_name=_string,
        batter=_integer,
        pitcher=_integer,
        events=_string,
        description=_string,
        spin_dir=_float,
        spin_rate=_float,
        break_angle=_float,
        break_length=_float,
        zone=_integer,
        des=_string,
        game_type=_string,
        stand=_string,
        p_throws=_string,
        home_team=_string,
        away_team=_string,
        type=_string,
        hit_location=_integer,
        bb_type=_integer,
        balls=_integer,
        strikes=_integer,
        game_year=_integer,
        pfx_x=_float,
        pfx_z=_float,
        px=_float,
        pz=_float,
        on_3b=_integer,
        on_2b=_integer,
        on_1b=_integer,
        outs_when_up=_integer,
        inning=_integer,
        inning_topbot=_string,
        hc_x=_float,
        hc_y=_float,
        tfs=_integer,
        tfs_zulu=_string,
        catcher=_integer,
        umpire=_integer,
        sv_id=_string,
        vx0=_float,
        vy0=_float,
        vz0=_float,
        ax=_float,
        ay=_float,
        az=_float,
        sz_top=_float,
        sz_bot=_float,
        hit_distance_sc=_integer,
        hit_speed=_float,
        hit_angle=_float,
        effective_speed=_float,
        release_spin_rate=_float,
        release_extension=_float,
        game_pk=_integer)

    def _getItems(self, d):
        '''Doc string'''

        items = []
        itemKeys = []
        for v in _venues:
            for dummy in range(100):
                try:
                    data = pd.read_csv(_baseURL.format(date=d, venue=v),
                                       parse_dates=[2],
                                       na_values='null')
                except Exception as e:
                    self.logger.debug(
                        '{!r} occurred while trying to dowload {} {}.'.
                        format(e, v, d))
                    time.sleep(5)
                else:
                    if not data.empty:
                        game_pks = data.game_pk.unique()
                        itemKeys.extend(game_pks)
                        for game_pk in game_pks:
                            items.append(data.iloc[
                                data.game_pk.values == game_pk, :])
                    break
            else:
                self.logger.error(
                    'Unable to download {} {} after {} attempts.'.
                    format(v, d, dummy + 1))

        return (items, itemKeys)
