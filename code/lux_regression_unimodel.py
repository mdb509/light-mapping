import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class LuxRegression:
    def __init__(self, lux, uV, degree=2):
        """
        Initialize the polynomial regression model for Lux prediction.
        """
        self.lux = lux
        self.uV = uV
        self.scaler = StandardScaler()
        self.uV_scaled = self.scaler.fit_transform(uV.reshape(-1, 1))

        # Polynomial regression setup
        self.poly = PolynomialFeatures(degree=degree)
        X_poly = self.poly.fit_transform(self.uV_scaled)
        self.model = LinearRegression().fit(X_poly, self.lux)

        mse = mean_squared_error(self.lux, self.model.predict(X_poly))
        print(f"→ Polynomial degree: {degree} (MSE = {mse:.2f})")

    def predict(self, uV_values):
        """
        Predict Lux from new µV values.
        """
        uV_values = np.array(uV_values).reshape(-1, 1)
        uV_scaled = self.scaler.transform(uV_values)
        X_poly = self.poly.transform(uV_scaled)
        return self.model.predict(X_poly)

    def evaluate(self, uV_input, lux_true=None, filename="plot.png"):
        """
        Evaluate and plot the model predictions.
        """
        uV_input = np.array(uV_input)
        lux_pred = self.predict(uV_input)

        if lux_true is not None:
            lux_true = np.array(lux_true)
            mse = mean_squared_error(lux_true, lux_pred)
        else:
            mse = None

        # Plot data
        plt.figure()
        plt.scatter(self.uV, self.lux, color='blue', label='Training Data')

        # Plot polynomial fit line
        uV_range = np.linspace(min(self.uV), max(self.uV), 300).reshape(-1, 1)
        uV_range_scaled = self.scaler.transform(uV_range)
        y_fit = self.model.predict(self.poly.transform(uV_range_scaled))
        plt.plot(uV_range, y_fit, color='red', label='Polynomial Fit')

        # Predicted points
        plt.scatter(uV_input, lux_pred, color='orange', marker='x', label='Predictions')

        # Add MSE text
        if mse is not None:
            plt.text(0.05, 0.95, f"MSE: {mse:.2f}", transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle="round", alpha=0.1))

        plt.title("Lux Prediction from µV")
        plt.xlabel("µV")
        plt.ylabel("Lux")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        print(f"Saved plot to: {filename}")
        if mse is not None:
            print(f"MSE: {mse:.2f}")
        return mse


# ========== MAIN ==========
if __name__ == "__main__":
    # Training data
    lux = np.array([91, 195, 426, 695, 997, 1440, 1738, 2343, 3006,
                    4110, 4690, 5200, 8070, 18970, 45300])
    uV = np.array([11148, 24289, 52238, 84773, 121109, 172078, 204312,
                   250687, 279031, 308203, 317750, 324984, 350780,
                   400269, 444031])

    # Initialize and evaluate model
    d = 13
    model = LuxRegression(lux, uV, degree=d)
    model.evaluate(uV_input=uV, lux_true=lux, filename=f"regression_train{d}.png")

    # Test predictions on new range
    test_uV = np.arange(1, 480000, 1000)
    predicted_lux = model.predict(test_uV)
    model.evaluate(uV_input=test_uV, lux_true=predicted_lux, filename="regression_test.png")
