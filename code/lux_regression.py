import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Manuell extrahierte Daten aus dem Bild
lux = np.array([
    91, 195, 426, 695, 997, 1440, 1738, 2343,
    3006, 4110, 4690, 5200, 8070, 18970, 45300
])
uV = np.array([
    11148, 24289, 52238, 84773, 121109, 172078, 204312, 250687,
    279031, 308203, 317750, 324984, 350780, 400269, 444031
])

# Input (X) = Mikrovolt, Output (y) = Lux
X = uV.reshape(-1, 1)
y = lux

# Polynom-Features erzeugen (z.B. Grad 3)
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# Regressionsmodell trainieren
model = LinearRegression()plt.savefig("lux_regression_plot.png")

model.fit(X_poly, y)

# Vorhersagen
y_pred = model.predict(X_poly)

# Fehler ausgeben
print("Mean Squared Error:", mean_squared_error(y, y_pred))

# Plot
plt.scatter(uV, lux, color='blue', label='Messdaten')
x_fit = np.linspace(min(uV), max(uV), 300).reshape(-1, 1)
x_fit_poly = poly.transform(x_fit)
y_fit = model.predict(x_fit_poly)
plt.plot(x_fit, y_fit, color='red', label='Regressionsmodell')
plt.xlabel("µV")
plt.ylabel("Lux")
plt.title("Regression µV -> Lux")
plt.legend()
plt.grid(True)
plt.savefig("lux_regression_plot.png")
