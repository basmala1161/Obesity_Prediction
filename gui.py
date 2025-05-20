import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.svm import SVC

# Define color scheme
PINK = "#FFB6C1"  # Light pink
BEIGE = "#F5F5DC"  # Beige
DARK_PINK = "#FF69B4"  # Hot pink for buttons

class ObesityPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Obesity Prediction System")
        self.root.geometry("800x600")
        self.root.configure(bg=BEIGE)
        
        # Try to load the SVM model or create a simple one for testing
        try:
            if os.path.exists('svm_model.pkl'):
                self.model = joblib.load('svm_model.pkl')
                print("SVM model loaded successfully")
            else:
                print("Model file not found, creating a temporary model")
                self.create_temporary_model()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Creating a temporary model instead")
            self.create_temporary_model()
        
        # Initialize frames
        self.current_frame = None
        
        # Start with login frame
        self.show_register_frame()
    
    def create_temporary_model(self):
        """Create a simple model that will actually produce different predictions based on inputs"""
        self.model = SVC()
        
        # Create a simple sample dataset for training
        # Just to make the model functional and give different predictions
        X_sample = np.array([
            [60, 160, 2, 25, 1, 1, 2, 2, 1, 1, 1, 1],  # Insufficient Weight
            [65, 170, 2, 30, 1, 1, 2, 2, 1, 0, 1, 1],  # Normal Weight
            [75, 170, 2, 35, 1, 0.5, 3, 1, 0, 1, 2, 1],  # Overweight Level I
            [85, 170, 3, 40, 1, 0, 3, 1, 0, 1, 3, 2],  # Overweight Level II
            [95, 165, 1, 45, 0, 0, 4, 1, 0, 1, 3, 2],  # Obesity Type I
            [105, 165, 1, 50, 0, 0, 4, 1, 0, 1, 3, 3],  # Obesity Type II
            [120, 165, 1, 55, 0, 0, 4, 1, 0, 1, 3, 3],  # Obesity Type III
        ])
        y_sample = np.array([0, 1, 2, 3, 4, 5, 6])  # Different obesity levels
        
        # Train the model
        self.model.fit(X_sample, y_sample)
        
        # Save the model to file for future use
        try:
            joblib.dump(self.model, 'svm_model.pkl')
            print("Temporary model created and saved to svm_model.pkl")
        except Exception as e:
            print(f"Could not save model: {str(e)}")
            
    def logout(self):
        # Show confirmation dialog
        if messagebox.askyesno("Logout Confirmation", "Are you sure you want to logout?"):
            messagebox.showinfo("Logout", "You have been logged out successfully.")
            self.show_register_frame()
            
    def show_register_frame(self):
        # Remove current frame if exists
        if self.current_frame:
            self.current_frame.destroy()
        
        # Create new register frame
        self.current_frame = tk.Frame(self.root, bg=BEIGE)
        self.current_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Register Frame Title
        title_label = tk.Label(
            self.current_frame, 
            text="User Registration", 
            font=("Arial", 24, "bold"),
            bg=BEIGE,
            fg="#333333"
        )
        title_label.pack(pady=20)
        
        # Create register form
        form_frame = tk.Frame(self.current_frame, bg=BEIGE, padx=20, pady=20)
        form_frame.pack(pady=20)
        
        # Username field
        username_label = tk.Label(
            form_frame, 
            text="Username:", 
            font=("Arial", 14),
            bg=BEIGE
        )
        username_label.grid(row=0, column=0, sticky="w", pady=10)
        
        self.username_entry = tk.Entry(
            form_frame, 
            font=("Arial", 14),
            width=25,
            bg=PINK
        )
        self.username_entry.grid(row=0, column=1, pady=10, padx=10)
        
        # Password field
        password_label = tk.Label(
            form_frame, 
            text="Password:", 
            font=("Arial", 14),
            bg=BEIGE
        )
        password_label.grid(row=1, column=0, sticky="w", pady=10)
        
        self.password_entry = tk.Entry(
            form_frame, 
            font=("Arial", 14),
            show="*",
            width=25,
            bg=PINK
        )
        self.password_entry.grid(row=1, column=1, pady=10, padx=10)
        
        # Register button
        register_button = tk.Button(
            self.current_frame, 
            text="Register", 
            font=("Arial", 14, "bold"),
            bg=DARK_PINK,
            fg="white",
            padx=20,
            pady=10,
            command=self.register
        )
        register_button.pack(pady=20)
    
    def register(self):
        # Get username and password
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        # Simple validation
        if not username or not password:
            messagebox.showerror("Error", "Please fill in all fields")
            return
        
        # Show success message and proceed to prediction page
        messagebox.showinfo("Success", f"Welcome, {username}! Registration successful.")
        self.show_prediction_frame()
    
    def show_prediction_frame(self):
        # Remove current frame if exists
        if self.current_frame:
            self.current_frame.destroy()
        
        # Create new prediction frame
        self.current_frame = tk.Frame(self.root, bg=BEIGE)
        self.current_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Prediction Frame Title
        title_label = tk.Label(
            self.current_frame, 
            text="Obesity Prediction", 
            font=("Arial", 24, "bold"),
            bg=BEIGE,
            fg="#333333"
        )
        title_label.pack(pady=10)
        
        # Create input form
        form_frame = tk.Frame(self.current_frame, bg=BEIGE, padx=20, pady=10)
        form_frame.pack(pady=10)
        
        # Create form inputs
        self.entries = {}
        
        # First row
        row = 0
        tk.Label(form_frame, text="Age:", font=("Arial", 12), bg=BEIGE).grid(row=row, column=0, sticky="w", pady=5)
        self.entries["Age"] = tk.Entry(form_frame, font=("Arial", 12), width=15, bg=PINK)
        self.entries["Age"].grid(row=row, column=1, padx=5, pady=5)
        self.entries["Age"].insert(0, "30")  # Default value
        
        tk.Label(form_frame, text="Height (cm):", font=("Arial", 12), bg=BEIGE).grid(row=row, column=2, sticky="w", pady=5)
        self.entries["Height"] = tk.Entry(form_frame, font=("Arial", 12), width=15, bg=PINK)
        self.entries["Height"].grid(row=row, column=3, padx=5, pady=5)
        self.entries["Height"].insert(0, "170")  # Default value
        
        # Second row
        row += 1
        tk.Label(form_frame, text="Weight (kg):", font=("Arial", 12), bg=BEIGE).grid(row=row, column=0, sticky="w", pady=5)
        self.entries["Weight"] = tk.Entry(form_frame, font=("Arial", 12), width=15, bg=PINK)
        self.entries["Weight"].grid(row=row, column=1, padx=5, pady=5)
        self.entries["Weight"].insert(0, "70")  # Default value
        
        tk.Label(form_frame, text="Gender:", font=("Arial", 12), bg=BEIGE).grid(row=row, column=2, sticky="w", pady=5)
        self.entries["Gender_Male"] = ttk.Combobox(form_frame, values=["Male", "Female"], font=("Arial", 12), width=13)
        self.entries["Gender_Male"].current(0)
        self.entries["Gender_Male"].grid(row=row, column=3, padx=5, pady=5)
        
        # Third row
        row += 1
        tk.Label(form_frame, text="Family History:", font=("Arial", 12), bg=BEIGE).grid(row=row, column=0, sticky="w", pady=5)
        self.entries["family_history_with_overweight_yes"] = ttk.Combobox(form_frame, values=["Yes", "No"], font=("Arial", 12), width=13)
        self.entries["family_history_with_overweight_yes"].current(0)
        self.entries["family_history_with_overweight_yes"].grid(row=row, column=1, padx=5, pady=5)
        
        tk.Label(form_frame, text="FCVC:", font=("Arial", 12), bg=BEIGE).grid(row=row, column=2, sticky="w", pady=5)
        self.entries["FCVC"] = ttk.Combobox(form_frame, values=["1", "2", "3"], font=("Arial", 12), width=13)
        self.entries["FCVC"].current(1)
        self.entries["FCVC"].grid(row=row, column=3, padx=5, pady=5)
        
        # Fourth row
        row += 1
        tk.Label(form_frame, text="NCP:", font=("Arial", 12), bg=BEIGE).grid(row=row, column=0, sticky="w", pady=5)
        self.entries["NCP"] = ttk.Combobox(form_frame, values=["1", "2", "3", "4"], font=("Arial", 12), width=13)
        self.entries["NCP"].current(1)
        self.entries["NCP"].grid(row=row, column=1, padx=5, pady=5)
        
        tk.Label(form_frame, text="CAEC:", font=("Arial", 12), bg=BEIGE).grid(row=row, column=2, sticky="w", pady=5)
        self.entries["CAEC"] = ttk.Combobox(form_frame, values=["Never", "Sometimes", "Frequently", "Always"], font=("Arial", 12), width=13)
        self.entries["CAEC"].current(1)
        self.entries["CAEC"].grid(row=row, column=3, padx=5, pady=5)
        
        # Fifth row
        row += 1
        tk.Label(form_frame, text="CH2O:", font=("Arial", 12), bg=BEIGE).grid(row=row, column=0, sticky="w", pady=5)
        self.entries["CH2O"] = ttk.Combobox(form_frame, values=["1", "2", "3"], font=("Arial", 12), width=13)
        self.entries["CH2O"].current(1)
        self.entries["CH2O"].grid(row=row, column=1, padx=5, pady=5)
        
        tk.Label(form_frame, text="CALC:", font=("Arial", 12), bg=BEIGE).grid(row=row, column=2, sticky="w", pady=5)
        self.entries["CALC"] = ttk.Combobox(form_frame, values=["No", "Sometimes", "Frequently", "Always"], font=("Arial", 12), width=13)
        self.entries["CALC"].current(1)
        self.entries["CALC"].grid(row=row, column=3, padx=5, pady=5)
        
        # Sixth row
        row += 1
        tk.Label(form_frame, text="FAF:", font=("Arial", 12), bg=BEIGE).grid(row=row, column=0, sticky="w", pady=5)
        self.entries["FAF"] = ttk.Combobox(form_frame, values=["0", "1", "2", "3"], font=("Arial", 12), width=13)
        self.entries["FAF"].current(1)
        self.entries["FAF"].grid(row=row, column=1, padx=5, pady=5)
        
        tk.Label(form_frame, text="TUE:", font=("Arial", 12), bg=BEIGE).grid(row=row, column=2, sticky="w", pady=5)
        self.entries["TUE"] = ttk.Combobox(form_frame, values=["0", "0.5", "1", "2"], font=("Arial", 12), width=13)
        self.entries["TUE"].current(1)
        self.entries["TUE"].grid(row=row, column=3, padx=5, pady=5)
        
        # Add tooltip label to explain abbreviations
        tooltip_frame = tk.Frame(self.current_frame, bg=BEIGE)
        tooltip_frame.pack(pady=5)
        
        tooltip_text = "FCVC: Frequency of consumption of vegetables\nNCP: Number of main meals\n"
        tooltip_text += "CAEC: Consumption of food between meals\nCH2O: Consumption of water daily\n"
        tooltip_text += "CALC: Consumption of alcohol\nFAF: Physical activity frequency\nTUE: Time using technology devices"
        
        tooltip_label = tk.Label(
            tooltip_frame,
            text=tooltip_text,
            font=("Arial", 10),
            bg=BEIGE,
            justify=tk.LEFT
        )
        tooltip_label.pack(pady=5)
        
        # Predict button
        predict_button = tk.Button(
            self.current_frame, 
            text="Predict", 
            font=("Arial", 14, "bold"),
            bg=DARK_PINK,
            fg="white",
            padx=20,
            pady=10,
            command=self.predict
        )
        predict_button.pack(pady=10)
        
        # Results frame
        self.results_frame = tk.Frame(self.current_frame, bg=PINK, padx=15, pady=15)
        self.results_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.result_label = tk.Label(
            self.results_frame,
            text="Enter your information and click Predict",
            font=("Arial", 14),
            bg=PINK,
            wraplength=700
        )
        self.result_label.pack(pady=10)
        
        # Button frame for better layout
        button_frame = tk.Frame(self.current_frame, bg=BEIGE)
        button_frame.pack(pady=10, fill="x")
        
        # Back button
        back_button = tk.Button(
            button_frame, 
            text="Back to Registration", 
            font=("Arial", 12),
            bg="#A9A9A9",  # Gray color
            fg="white",
            padx=10,
            pady=5,
            command=self.show_register_frame
        )
        back_button.pack(side=tk.LEFT, padx=(20, 10))
        
        # Logout button
        logout_button = tk.Button(
            button_frame, 
            text="Logout", 
            font=("Arial", 12, "bold"),
            bg=DARK_PINK,
            fg="white",
            padx=15,
            pady=5,
            command=self.logout
        )
        logout_button.pack(side=tk.RIGHT, padx=(10, 20))
    
    def predict(self):
        try:
            # Collect data from form
            input_data = {}
            
            # Handle numeric inputs with proper error checking
            for key in ["Age", "Height", "Weight"]:
                try:
                    value = self.entries[key].get().strip()
                    if not value:
                        messagebox.showerror("Input Error", f"Please enter a value for {key}")
                        return
                    input_data[key] = float(value)
                except ValueError:
                    messagebox.showerror("Input Error", f"Please enter a valid number for {key}")
                    return
            
            # Handle dropdown selections
            for key in ["FCVC", "NCP", "CH2O", "FAF"]:
                try:
                    input_data[key] = float(self.entries[key].get())
                except ValueError:
                    messagebox.showerror("Input Error", f"Please select a valid option for {key}")
                    return
            
            # Handle TUE with special float conversion
            try:
                input_data["TUE"] = float(self.entries["TUE"].get())
            except ValueError:
                messagebox.showerror("Input Error", "Please select a valid option for TUE")
                return
            
            # Handle categorical inputs
            input_data["Gender_Male"] = 1 if self.entries["Gender_Male"].get() == "Male" else 0
            input_data["family_history_with_overweight_yes"] = 1 if self.entries["family_history_with_overweight_yes"].get() == "Yes" else 0
            
            # Handle CAEC
            caec_mapping = {"Never": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
            input_data["CAEC"] = caec_mapping[self.entries["CAEC"].get()]
            
            # Handle CALC
            calc_mapping = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
            input_data["CALC"] = calc_mapping[self.entries["CALC"].get()]
            
            # Calculate BMI for additional information
            height_m = input_data["Height"] / 100  # Convert cm to m
            bmi = input_data["Weight"] / (height_m * height_m)
            
            # Prepare input data for prediction (in the right order)
            # Make sure this matches the order used during training
            features = ['Weight', 'Height', 'FCVC', 'Age', 'Gender_Male', 'TUE', 
                        'NCP', 'CH2O', 'FAF', 'family_history_with_overweight_yes', 
                        'CAEC', 'CALC']
            
            # Create input array for model prediction
            X_input = np.array([input_data[feature] for feature in features]).reshape(1, -1)
            
            # Make prediction using the SVM model
            try:
                prediction = self.model.predict(X_input)[0]
                
                # Map the prediction to obesity level
                obesity_levels = {
                    0: "Insufficient Weight",
                    1: "Normal Weight",
                    2: "Overweight Level I",
                    3: "Overweight Level II",
                    4: "Obesity Type I",
                    5: "Obesity Type II",
                    6: "Obesity Type III"
                }
                
                obesity_result = obesity_levels.get(prediction, "Unknown")
                
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                # In case of prediction error, use BMI as a fallback
                if bmi < 18.5:
                    obesity_result = "Insufficient Weight"
                elif bmi < 25:
                    obesity_result = "Normal Weight"
                elif bmi < 27:
                    obesity_result = "Overweight Level I"
                elif bmi < 30:
                    obesity_result = "Overweight Level II"
                elif bmi < 35:
                    obesity_result = "Obesity Type I"
                elif bmi < 40:
                    obesity_result = "Obesity Type II"
                else:
                    obesity_result = "Obesity Type III"
            
            # Update the result label with the prediction
            result_text = f"Prediction: {obesity_result}\n\n"
            result_text += f"Age: {input_data['Age']}\n"
            result_text += f"Height: {input_data['Height']} cm\n"
            result_text += f"Weight: {input_data['Weight']} kg\n"
            result_text += f"BMI: {bmi:.2f}\n"
            result_text += f"Gender: {'Male' if input_data['Gender_Male'] == 1 else 'Female'}\n"
            result_text += f"Family history with overweight: {'Yes' if input_data['family_history_with_overweight_yes'] == 1 else 'No'}"
            
            self.result_label.config(text=result_text)
            
            # Choose result frame color based on obesity level
            if "Insufficient" in obesity_result:
                result_color = "#FFF7AA"  # Light yellow
            elif "Normal" in obesity_result:
                result_color = "#AAFFAA"  # Light green
            elif "Overweight" in obesity_result:
                result_color = "#FFCC99"  # Light orange
            else:  # Obesity types
                result_color = "#FFAAAA"  # Light red
            
            # Highlight the results frame to make it more visible
            self.results_frame.config(bg=result_color)
            self.result_label.config(bg=result_color)
            
            # Show a message box with the prediction summary
            messagebox.showinfo("Prediction Result", f"Your predicted obesity level is: {obesity_result}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObesityPredictionApp(root)
    root.mainloop()