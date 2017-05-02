# %% Imports

import pandas as pd
from matplotlib import pyplot as plt

from statcast.plot import plotMLBLogos

# %% Plot 2002 Salary vs Wins

plt.style.use('blackontrans')

df = pd.read_csv('/Users/mattfay/Downloads/MLB02.csv',
                 names=['team', 'winP', 'money'], index_col=0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plotMLBLogos(df.money * 1e-6, df.winP * 1e2, sizes=35, ax=ax)

ax.set_xlabel('Salary (Million $)')
ax.set_ylabel('Winning Percentage (%)')
ax.set_title('2002 MLB Regular Season')
ax.set_ybound(ax.get_ylim()[0] - 1.2, ax.get_ylim()[1] + 1.2)

fig.savefig('MLB02.png')
