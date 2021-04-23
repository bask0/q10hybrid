
import pandas as pd
import xarray as xr

d = pd.read_csv('/Users/bk/Files/usmile/data/q10/Synthetic4BookChap.csv')[
    ['SW_POT_sm', 'SW_POT_sm_diff', 'TA', 'RECO_syn', 'Rb_syn']]
d = d.rename(columns={
    'SW_POT_sm': 'sw_pot',
    'SW_POT_sm_diff': 'dsw_pot',
    'TA': 'ta',
    'RECO_syn': 'reco',
    'Rb_syn': 'rb'}
)
d['time'] = pd.date_range('2002-01-01 00:15', '2012-12-31 23:45', freq='30min')

d = xr.Dataset.from_dataframe(d.set_index('time')).sel(time=slice('2003-01-01', '2011-12-31'))
d.to_netcdf('/Users/bk/Files/usmile/data/q10/Synthetic4BookChap.nc')
