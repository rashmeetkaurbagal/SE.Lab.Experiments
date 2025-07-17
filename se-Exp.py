import numpy as np
import matplotlib.pyplot as plt

# Step 1: Input Data (Synthetic Real-Time-ish Example)

time_hours = np.array([0, 4, 8, 12, 16, 20])
humidity_percent = np.array([85, 80, 60, 55, 65, 75])
rainfall_mm = np.array([10, 5, 2, 0, 3, 7])  # mm of rainfall

# Step 2: Fit Quadratic Models for Humidity and Rainfall
humidity_coeffs = np.polyfit(time_hours, humidity_percent, 2)
rainfall_coeffs = np.polyfit(time_hours, rainfall_mm, 2)

print(f"Humidity Quadratic Model: {humidity_coeffs[0]:.4f}t² + {humidity_coeffs[1]:.4f}t + {humidity_coeffs[2]:.4f}")
print(f"Rainfall Quadratic Model: {rainfall_coeffs[0]:.4f}t² + {rainfall_coeffs[1]:.4f}t + {rainfall_coeffs[2]:.4f}")

# Step 3: Predict values for every hour
t_values = np.arange(0, 25, 1)
predicted_humidity = np.polyval(humidity_coeffs, t_values)
predicted_rainfall = np.polyval(rainfall_coeffs, t_values)

# Step 4: Plot Graphs

plt.figure(figsize=(14, 6))

# Humidity Plot
plt.subplot(1, 2, 1)
plt.scatter(time_hours, humidity_percent, color='blue', label='Observed Humidity')
plt.plot(t_values, predicted_humidity, color='red', linestyle='--', label='Predicted Humidity')
plt.title('Humidity Prediction using Quadratic Model')
plt.xlabel('Time (Hours)')
plt.ylabel('Humidity (%)')
plt.grid(True)
plt.legend()

# Rainfall Plot
plt.subplot(1, 2, 2)
plt.scatter(time_hours, rainfall_mm, color='green', label='Observed Rainfall')
plt.plot(t_values, predicted_rainfall, color='purple', linestyle='--', label='Predicted Rainfall')
plt.title('Rainfall Prediction using Quadratic Model')
plt.xlabel('Time (Hours)')
plt.ylabel('Rainfall (mm)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
