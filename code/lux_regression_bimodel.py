import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class LuxRegression:
    def __init__(self, lux, uV, lux_threshold=1500, degree=5):
        """
        Initialize the regression model.
        """
        self.lux_threshold = lux_threshold
        self.lux = lux
        self.uV = uV
        self.scaler = StandardScaler()
        uV_scaled = self.scaler.fit_transform(uV.reshape(-1, 1))

        self.uV_low = uV_scaled[lux <= lux_threshold]
        self.lux_low = lux[lux <= lux_threshold]
        self.uV_high = uV_scaled[lux > lux_threshold]
        self.lux_high = lux[lux > lux_threshold]

        # Linear model
        self.model_low = LinearRegression().fit(self.uV_low, self.lux_low)

        # Polynomial model for non linear part
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(self.uV_high)
        model = LinearRegression().fit(X_poly, self.lux_high)
        self.poly_high = poly
        self.model_high = model

    def predict(self, uV_values):
        """ predict values from µV to lux"""
        uV_values = np.array(uV_values).reshape(-1, 1)
        uV_scaled = self.scaler.transform(uV_values)
        predictions = []
        
        # split for linear and non linear parts 
        for u in uV_scaled:
            pred_low = self.model_low.predict([u])[0]
            if pred_low <= self.lux_threshold:
                predictions.append(pred_low)
            else:
                poly_feat = self.poly_high.transform([u])
                predictions.append(self.model_high.predict(poly_feat)[0])

        return np.array(predictions)

    def evaluate(self, uV_input, lux_true=None, filename="plot.png"):
        """ evaluate predicton calculate mse and save plot"""
        uV_input = np.array(uV_input)
        lux_pred = self.predict(uV_input)

        if lux_true is not None:
            lux_true = np.array(lux_true)
            mask_low = lux_true <= self.lux_threshold
            mask_high = ~mask_low
            mse_low = mean_squared_error(lux_true[mask_low], lux_pred[mask_low]) if mask_low.any() else None
            mse_high = mean_squared_error(lux_true[mask_high], lux_pred[mask_high]) if mask_high.any() else None
        else:
            mse_low = mse_high = None

        plt.figure()
        plt.scatter(self.uV, self.lux, color='blue', label='Original Data')

        # Plot linear fit
        x_low = np.linspace(self.uV[self.lux <= self.lux_threshold].min(),
                            self.uV[self.lux <= self.lux_threshold].max(), 300).reshape(-1, 1)
        x_low_scaled = self.scaler.transform(x_low)
        y_low = self.model_low.predict(x_low_scaled)
        plt.plot(x_low, y_low, color='red', label='Linear Fit (Low Lux)')

        # Plot polynomial fit
        x_high = np.linspace(self.uV[self.lux > self.lux_threshold].min(),
                             self.uV[self.lux > self.lux_threshold].max(), 300).reshape(-1, 1)
        x_high_scaled = self.scaler.transform(x_high)
        y_high = self.model_high.predict(self.poly_high.transform(x_high_scaled))
        plt.plot(x_high, y_high, color='green', label='Polynomial Fit (High Lux)')

        plt.scatter(uV_input, lux_pred, color='orange', marker='x', label='Predictions')

        text = ''
        if mse_low is not None:
            text += f"MSE ≤ {self.lux_threshold}: {mse_low:.2f}\n"
        if mse_high is not None:
            text += f"MSE > {self.lux_threshold}: {mse_high:.2f}"
        if text:
            plt.text(0.05, 0.95, text.strip(), transform=plt.gca().transAxes,
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
        if mse_low is not None:
            print(f"MSE (≤ {self.lux_threshold}): {mse_low:.2f}")
        if mse_high is not None:
            print(f"MSE (> {self.lux_threshold}): {mse_high:.2f}")
        return mse_low, mse_high


# ========== MAIN ==========
if __name__ == "__main__":
    # train data
    lux = np.array([91, 195, 426, 695, 997, 1440, 1738, 2343, 3006, 4110, 4690, 5200, 8070, 18970, 45300])
    uV = np.array([11148, 24289, 52238, 84773, 121109, 172078, 204312, 250687, 279031,
                   308203, 317750, 324984, 350780, 400269, 444031])

    # evaluate train data
    model = LuxRegression(lux, uV)
    model.evaluate(uV_input=uV, lux_true=lux, filename="regression_train.png")

    # evaluate test data
    test_uV = np.arange(1, 210000, 1000)
    fake_lux_true, _ = model.predict(test_uV), None
    model.evaluate(uV_input=test_uV, lux_true=fake_lux_true, filename="regression_test.png")
