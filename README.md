## Devloped by: DAKSHATA G
## Register Number: 212223240021
## Date: 16-05-26

# Ex.No:04           FIT ARMA MODEL FOR TIME SERIES

### AIM:
To implement ARMA model in python.

### ALGORITHM:
1. Import necessary libraries.

2. Set up matplotlib settings for figure size.

3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.

5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.


### PROGRAM:

Import necessary Modules and Functions
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```
Load dataset
```py
data=pd.read_csv('/content/AirPassengers.csv')
```
Declare required variables and set figure size, and visualise the data
```py
N=1000
plt.rcParams['figure.figsize'] = [12, 6] #plt.rcParams is a dictionary-like object in Matplotlib that stores global settings for plots. The "rc" in rcParams stands for runtime configuration. It allows you to customize default styles for figures, fonts, colors, sizes, and more.

X=data['#Passengers']
plt.plot(X)
plt.title('Original Data')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()
```
Fitting the ARMA(1,1) model and deriving parameters
```py
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']
```
Simulate ARMA(1,1) Process
```py
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
```
Plot ACF and PACF for ARMA(1,1)
```py
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()
```
Fitting the ARMA(1,1) model and deriving parameters
```py
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
```
Simulate ARMA(2,2) Process
```py
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])  
ma2 = np.array([1, theta1_arma22, theta2_arma22])  
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()
```
Plot ACF and PACF for ARMA(2,2)
```py
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()

```

### OUTPUT:

Original data:

<img width="1083" height="589" alt="image" src="https://github.com/user-attachments/assets/631614ed-ab19-4088-ae2e-d4246a6686f8" />


Partial Autocorrelation:-

<img width="1067" height="307" alt="image" src="https://github.com/user-attachments/assets/b8425c14-f3bc-4d5d-935e-270f8c2701e3" />


Autocorrelation:-

<img width="1069" height="284" alt="image" src="https://github.com/user-attachments/assets/fe045587-1d22-4165-99c0-e69126c96b32" />


SIMULATED ARMA(1,1) PROCESS:

<img width="1092" height="573" alt="image" src="https://github.com/user-attachments/assets/556b577f-38c5-454a-b524-eb3aa91074d5" />


Partial Autocorrelation:-

<img width="1053" height="558" alt="image" src="https://github.com/user-attachments/assets/de3c0274-a606-4bce-a78c-966a337cf6c6" />


Autocorrelation:-

<img width="1082" height="577" alt="image" src="https://github.com/user-attachments/assets/38e19fe0-6c6b-4d56-805c-c95e0e56edf4" />



SIMULATED ARMA(2,2) PROCESS:

<img width="1086" height="597" alt="image" src="https://github.com/user-attachments/assets/4558e973-0079-42e5-9b3a-d253bdd45a58" />


Partial Autocorrelation

<img width="1063" height="576" alt="image" src="https://github.com/user-attachments/assets/b09f31c7-8d43-4e58-b0cd-7440a41fca2e" />


Autocorrelation
<img width="1073" height="597" alt="image" src="https://github.com/user-attachments/assets/8416b917-3408-47ee-950b-47afac8a8c2f" />


### RESULT:

Thus, a python program is created to fit ARMA Model successfully.
