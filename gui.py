import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("stack_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Features with descriptions
features_with_descriptions = {
    'Weight': 'Weight (kg)',
    'Height': 'Height (cm)',
    'FCVC': 'Vegetable consumption (1-3)',
    'Age': 'Age (years)',
    'Gender_Male': 'Gender (1 = Male, 0 = Female)',
    'TUE': 'Technology use (0-3)',
    'NCP': 'Main meals/day (1-5)',
    'CH2O': 'Water intake (1-3)',
    'FAF': 'Physical activity (0-3)',
    'family_history_with_overweight_yes': 'Family history (1 = Yes, 0 = No)',
    'CAEC': 'Snacking (0=Never → 3=Always)',
    'CALC': 'Alcohol (0=Never → 3=Always)'
}

# GUI setup
root = tk.Tk()
root.title("Obesity Level Prediction")
root.geometry("800x850")
root.configure(bg="#ffffff")  # White background

# Title
title_label = tk.Label(
    root,
    text="OBESITY LEVEL PREDICTION",
    font=("Arial", 22, "bold"),
    bg="#ffffff",
    fg="#b00000"  # Red
)
title_label.pack(pady=20)

# Description
desc_label = tk.Label(
    root,
    text="Enter your health and lifestyle information below to predict your obesity level.",
    font=("Arial", 11, "bold"),
    bg="#ffffff",
    fg="#000000"  # Black
)
desc_label.pack()

separator = tk.Label(root, text="―" * 100, bg="#ffffff", fg="#cccccc")
separator.pack(pady=10)

# Input fields
entries = {}
input_frame = tk.Frame(root, bg="#ffffff")
input_frame.pack(pady=10)

for feature, description in features_with_descriptions.items():
    frame = tk.Frame(input_frame, bg="#ffffff")
    frame.pack(pady=8, fill='x')

    label = tk.Label(frame, text=description + ":", anchor='w', width=45,
                     bg="#ffffff", fg="#000000", font=("Arial", 10, "bold"))
    label.pack(side=tk.LEFT, padx=12)

    entry = tk.Entry(frame, width=25, font=("Arial", 11))
    entry.pack(side=tk.LEFT)
    entries[feature] = entry

# Predict function
def predict():
    try:
        input_data = []
        for feat, desc in features_with_descriptions.items():
            val = entries[feat].get().strip()
            if val == "":
                raise ValueError(f"Please enter a value for: {desc}")
            num_val = float(val)

            # Validation
            if feat == 'Weight' and not (30 <= num_val <= 200):
                raise ValueError("Weight must be between 30 and 200 kg.")
            if feat == 'Height' and not (100 <= num_val <= 220):
                raise ValueError("Height must be between 100 and 220 cm.")
            if feat in ['FCVC', 'CH2O'] and not (1 <= num_val <= 3):
                raise ValueError(f"{desc} must be between 1 and 3.")
            if feat in ['TUE', 'FAF'] and not (0 <= num_val <= 3):
                raise ValueError(f"{desc} must be between 0 and 3.")
            if feat == 'NCP' and not (1 <= num_val <= 5):
                raise ValueError("NCP must be between 1 and 5.")
            if feat == 'Age' and not (10 <= num_val <= 100):
                raise ValueError("Age must be between 10 and 100.")
            if feat in ['Gender_Male', 'family_history_with_overweight_yes'] and num_val not in [0, 1]:
                raise ValueError(f"{desc} must be 0 or 1.")
            if feat in ['CAEC', 'CALC'] and not (0 <= num_val <= 3):
                raise ValueError(f"{desc} must be between 0 and 3.")

            input_data.append(num_val)

        input_array = np.array(input_data).reshape(1, -1)
        prediction_index = model.predict(input_array)[0]
        prediction_label = label_encoder.inverse_transform([prediction_index])[0]

        messagebox.showinfo("Prediction Result", f"Predicted Obesity Level:\n\n{prediction_label}")

    except Exception as e:
        messagebox.showerror("Input Error", str(e))

# Reset
def reset_fields():
    for entry in entries.values():
        entry.delete(0, tk.END)

# Buttons
button_frame = tk.Frame(root, bg="#ffffff")
button_frame.pack(pady=30)

predict_btn = tk.Button(button_frame, text="Predict", command=predict,
                        bg="#b00000", fg="white", font=("Arial", 12, "bold"), width=16)
predict_btn.pack(side=tk.LEFT, padx=20)

reset_btn = tk.Button(button_frame, text="Reset", command=reset_fields,
                      bg="#444444", fg="white", font=("Arial", 12), width=16)
reset_btn.pack(side=tk.LEFT, padx=20)

# Run GUI
root.mainloop()
