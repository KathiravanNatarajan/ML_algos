import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate some example data (replace with your actual time series data)
np.random.seed(0)
dates = pd.date_range('2023-01-01', periods=365)
data = np.random.randn(365) + np.sin(np.linspace(0, 4*np.pi, 365))  # Example data with trend and seasonality
ts = pd.Series(data, index=dates)

# Decompose the time series
decomposition = seasonal_decompose(ts, model='additive')

# Extract the components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the components
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(ts, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

plt.show()
