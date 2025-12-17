import numpy as np
import yfinance as yf
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

tickers_list = ["AAPL", 'GOOGL', 'NVDA', 'TSLA', 'PFE', '^GSPC']
dt = pd.DataFrame(yf.download(tickers_list, period = '10y', interval='1d'))

irx = yf.download("^IRX", period='10y', interval='1d')
irx['daily_return'] = irx['Close'] / 100 / 252
irx = irx[['daily_return']]
irx.columns = ['IRX']

dt = dt.T.reset_index()
main = dt[dt['Price'] == 'Close']

main = main.drop(columns='Price')
main = main.set_index('Ticker')
main = main.T

returns = main[['AAPL', 'GOOGL', 'NVDA', 'TSLA', 'PFE', '^GSPC']].pct_change()
returns.columns = [f"{col}_dailyChange" for col in returns.columns]

main = pd.concat([returns, irx], axis=1)

summary = main.agg(['mean', 'std']).T

n_assets = len(tickers_list)
n_combs = 10000

weights = np.random.dirichlet(np.ones(n_assets), size=n_combs)
weights_df = pd.DataFrame(weights, columns=[f'w{i+1}' for i in range(n_assets)])
corners = pd.DataFrame({'w1':[1,0,0,0,0,0], 'w2': [0,1,0,0,0,0], 'w3':[0,0,1,0,0,0], 'w4': [0,0,0,1,0,0], 'w5': [0,0,0,0,1,0], 'w6': [0,0,0,0,0,1]})
weights_df = pd.concat([weights_df, corners], axis=0)

for i in range(1, 7):
    weights_df[f"return_{i}"] = summary.iloc[i-1]['mean']
    weights_df[f"risk_{i}"] = summary.iloc[i-1]['std']

def get_risk(row, matrix):
    weights = np.array(row[[f"w{i}" for i in range(1, n_assets+1)]]).reshape(-1,1)
    return weights.T @ matrix @ weights

cov_mat = main[[col for col in main.columns if col != 'IRX']].cov().to_numpy()

weights_df['p_return'] = weights_df.apply(lambda row: np.dot(np.array(row[[f"w{i}" for i in range(1, 7)]]), np.array(row[[f"return_{j}" for j in range(1, 7)]])), axis=1)
weights_df['p_risk'] = weights_df.apply(lambda row: (get_risk(row,cov_mat)[0][0])**0.5, axis=1)

rf_value = float(summary.loc['IRX']['mean'])

weights_df['sharpeRatio'] = (weights_df['p_return'] - rf_value)/weights_df['p_risk']
max_sharpe = weights_df[weights_df['sharpeRatio'] == weights_df['sharpeRatio'].max()]
max_sharpe_risk = max_sharpe['p_risk']
max_sharpe_return = max_sharpe['p_return']
max_sharpe_rat = max_sharpe['sharpeRatio']

points = summary.iloc[:-1][['std', 'mean']]
points = weights_df[['p_risk', 'p_return']]
points = points.to_numpy()
labels = summary.index[:-1]
hull = ConvexHull(points)

x = np.linspace(0, 0.04, 1000)
y = [rf_value + c*max_sharpe_rat for c in x]

example_port = [0.025, 0.0015]

x1 = [example_port[0]]*1000
y1 = np.linspace(0, example_port[1], 1000)
x2 = np.linspace(0, example_port[0], 1000)
y2 = [example_port[1]]*1000

def invert(c, m, y_value):
    return (y_value-c)/m

val2 = invert(rf_value, float(max_sharpe_rat), example_port[1])

x3 = [val2]*1000
y3 = np.linspace(0, example_port[1], 1000)

plt.figure(figsize=(12,6))
plt.scatter(summary['std'], summary['mean'])
plt.scatter(max_sharpe_risk, max_sharpe_return)
plt.scatter(example_port[0], example_port[1], marker='D', facecolors='none', edgecolors='black')
plt.plot(x, y, linestyle='-.', color='grey')
plt.fill(points[hull.vertices,0], points[hull.vertices,1], 'orange', alpha=0.3)
plt.plot(x1, y1, linestyle='--', color='grey')
plt.plot(x2, y2, linestyle='--', color='grey')
plt.plot(x3, y3, linestyle='--', color='grey')
plt.xlabel(r"Risk $\sigma$")
plt.ylabel(r"Return $E[R_p]$")

plt.savefig("/Users/jackadeney/Documents/sharpeRatioWithBias/main_graph.jpeg")

# 1. construct risk return convex hull of all possible portfolio allocations
# 2. find sharpe ratio maximizing portfolio
# 3. find portfolio RFxP line such that return == portfolio return
# 4. find Markowitz parameter lambda such that portfolio RFxP is maximal
# 5. use this measure to compare portfolio manager










