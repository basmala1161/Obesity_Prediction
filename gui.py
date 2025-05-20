import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

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
        
        # Initialize frames
        self.current_frame = None
        
        # Start with login frame
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
        
        # Information note
        info_frame = tk.Frame(self.current_frame, bg=PINK, padx=15, pady=10)
        info_frame.pack(fill="x", padx=20, pady=10)
        
        info_text = tk.Label(
            info_frame,
            text="Please register to use the Obesity Prediction System.\nYour information will be kept confidential and secure.",
            font=("Arial", 12),
            bg=PINK,
            justify="left"
        )
        info_text.pack(anchor="w")
        
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
        
        # Username note
        username_note = tk.Label(
            form_frame,
            text="Choose a username you can easily remember",
            font=("Arial", 10, "italic"),
            bg=BEIGE,
            fg="#666666"
        )
        username_note.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        # Password field
        password_label = tk.Label(
            form_frame, 
            text="Password:", 
            font=("Arial", 14),
            bg=BEIGE
        )
        password_label.grid(row=2, column=0, sticky="w", pady=10)
        
        self.password_entry = tk.Entry(
            form_frame, 
            font=("Arial", 14),
            show="*",
            width=25,
            bg=PINK
        )
        self.password_entry.grid(row=2, column=1, pady=10, padx=10)
        
        # Password note
        password_note = tk.Label(
            form_frame,
            text="For security, consider using a strong password",
            font=("Arial", 10, "italic"),
            bg=BEIGE,
            fg="#666666"
        )
        password_note.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
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
        # Get username and password (not doing anything with them as requested)
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        # Simple validation
        if not username or not password:
            messagebox.showerror("Error", "Please fill in all fields")
            return
        
        # Just show success message and proceed to prediction page
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
        
        # Add explanation frame at the top
        info_frame = tk.Frame(form_frame, bg=PINK, padx=10, pady=10)
        info_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=10)
        
        info_text = tk.Label(
            info_frame,
            text="Please fill in all fields below. Your information will be used to predict your obesity level and provide health suggestions.",
            font=("Arial", 11),
            bg=PINK,
            wraplength=700,
            justify="left"
        )
        info_text.pack(fill="both")
        
        # Create form inputs
        self.entries = {}
        
        # First row (after info frame)
        row = 1
        
        # Basic information section
        section_label = tk.Label(
            form_frame, 
            text="Basic Information:", 
            font=("Arial", 12, "bold"),
            bg=BEIGE
        )
        section_label.grid(row=row, column=0, columnspan=4, sticky="w", pady=5)
        
        row += 1
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
        
        # Add explanation for family history
        row += 1
        family_info_frame = tk.Frame(form_frame, bg="#FFD1DC", padx=5, pady=5)  # Lighter pink
        family_info_frame.grid(row=row, column=0, columnspan=4, sticky="ew", pady=5)
        
        family_info = tk.Label(
            family_info_frame,
            text="Family history refers to whether any blood relatives have or had weight-related issues.",
            font=("Arial", 10, "italic"),
            bg="#FFD1DC",
            wraplength=700
        )
        family_info.pack(anchor="w")
        
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
        
        # Add explanation for dietary habits
        row += 1
        diet_info_frame = tk.Frame(form_frame, bg="#FFD1DC", padx=5, pady=5)
        diet_info_frame.grid(row=row, column=0, columnspan=4, sticky="ew", pady=5)
        
        diet_info = tk.Label(
            diet_info_frame,
            text="FCVC: Frequency of consumption of vegetables (1=Never, 2=Sometimes, 3=Always)\nNCP: Number of main meals per day\nCAEC: Consumption of food between meals (snacking habits)",
            font=("Arial", 10, "italic"),
            bg="#FFD1DC",
            wraplength=700,
            justify="left"
        )
        diet_info.pack(anchor="w")
        
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
        
        # Add explanation for lifestyle habits
        row += 1
        lifestyle_info_frame = tk.Frame(form_frame, bg="#FFD1DC", padx=5, pady=5)
        lifestyle_info_frame.grid(row=row, column=0, columnspan=4, sticky="ew", pady=5)
        
        lifestyle_info = tk.Label(
            lifestyle_info_frame,
            text="CH2O: Daily water consumption (1=Less than 1L, 2=1-2L, 3=More than 2L)\nCALC: Consumption of alcohol (No/Sometimes/Frequently/Always)\nFAF: Physical activity frequency (0=None, 1=1-2 days, 2=3-4 days, 3=5+ days)\nTUE: Time using technology devices daily (in hours)",
            font=("Arial", 10, "italic"),
            bg="#FFD1DC",
            wraplength=700,
            justify="left"
        )
        lifestyle_info.pack(anchor="w")
        
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
        
        # Back button
        back_button = tk.Button(
            self.current_frame, 
            text="Back to Registration", 
            font=("Arial", 12),
            bg=DARK_PINK,
            fg="white",
            command=self.show_register_frame
        )
        back_button.pack(pady=10)
    
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
            
            # Check for valid height and weight
            if input_data["Height"] <= 0:
                messagebox.showerror("Input Error", "Height must be greater than 0")
                return
                
            if input_data["Weight"] <= 0:
                messagebox.showerror("Input Error", "Weight must be greater than 0")
                return
            
            # Handle dropdown selections
            for key in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
                try:
                    input_data[key] = float(self.entries[key].get())
                except ValueError:
                    messagebox.showerror("Input Error", f"Please select a valid option for {key}")
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
            
            # Calculate BMI for predictions
            height_m = input_data["Height"] / 100  # Convert cm to m
            bmi = input_data["Weight"] / (height_m ** 2)
            
            # Generate a prediction based on BMI
            if bmi < 18.5:
                obesity_level = "Insufficient Weight"
                suggestions = "Consider increasing caloric intake and consulting with a nutritionist."
            elif bmi < 25:
                obesity_level = "Normal Weight"
                suggestions = "Maintain your healthy lifestyle with balanced diet and regular exercise."
            elif bmi < 30:
                obesity_level = "Overweight"
                suggestions = "Consider more physical activity and reducing caloric intake slightly."
            elif bmi < 35:
                obesity_level = "Obesity Type I"
                suggestions = "Consult with healthcare professional. Focus on balanced diet and regular exercise."
            elif bmi < 40:
                obesity_level = "Obesity Type II"
                suggestions = "Medical consultation recommended. Consider structured weight management program."
            else:
                obesity_level = "Obesity Type III"
                suggestions = "Seek medical advice immediately. Professional weight management intervention needed."
            
            # Add factors that might adjust the prediction
            additional_factors = []
            if input_data["family_history_with_overweight_yes"] == 1:
                additional_factors.append("Family history of obesity increases risk.")
            
            if input_data["FAF"] <= 1:
                additional_factors.append("Low physical activity level may contribute to weight gain.")
            
            if input_data["CH2O"] <= 1:
                additional_factors.append("Insufficient water intake may affect metabolism.")
            
            if input_data["FCVC"] <= 1:
                additional_factors.append("Low vegetable consumption may affect nutritional balance.")
            
            # Personalized suggestions based on inputs
            personalized = []
            if input_data["CAEC"] >= 2:
                personalized.append("Consider reducing snacking between meals.")
            
            if input_data["TUE"] >= 1.5:
                personalized.append("Try to reduce screen time and increase physical activity.")
            
            if input_data["CALC"] >= 2:
                personalized.append("Reducing alcohol consumption could help with weight management.")
            
            # Combine all factors and suggestions
            factors_text = "\n".join(additional_factors) if additional_factors else "No additional risk factors identified."
            personalized_text = "\n".join(personalized) if personalized else "Continue with your current healthy habits."
            
            # Create the result text
            result_text = f"Prediction: {obesity_level}\n\nBMI: {bmi:.2f}\n\nFactors:\n{factors_text}\n\nSuggestions:\n{suggestions}\n\nPersonalized Recommendations:\n{personalized_text}"
            
            # Update the result label with the prediction
            self.result_label.config(text=result_text)
            
            # Highlight the results frame to make it more visible
            self.results_frame.config(bg="#FFA0B0")  # Slightly darker pink to draw attention
            
            # Show a message box with the prediction summary
            messagebox.showinfo("Prediction Result", f"Your predicted obesity level is: {obesity_level}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObesityPredictionApp(root)
    root.mainloop() 
    