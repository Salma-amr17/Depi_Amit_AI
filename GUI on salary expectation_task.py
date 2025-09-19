'''
GUI with Tkinter => Predict Salary from Experience
'''
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ======== Load and prepare the data ========
dataset = pd.read_csv(r'Salary_Data.csv')
X = dataset.iloc[:, :-1]   
Y = dataset.iloc[:, 1]     

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# ======== GUI ==========
app = tk.Tk()
app.title("Experience vs Salary Prediction")
app.geometry("420x220")
app.config(bg="#b3d9ff")  # light blue shade

lbl = tk.Label(app, text="Enter Years of Experience:", bg="#b3d9ff", fg="black", font=("Arial", 11))
lbl.pack(pady=12)

txt_input = tk.Entry(app, width=25)
txt_input.pack(pady=6)

def show_prediction():
    value = txt_input.get()   

    if not value:
        messagebox.showerror("Error", "Please enter a valid number!")
        return

    try:
        exp = float(value)
    except:
        messagebox.showerror("Error", "Only numeric values are allowed!")
        return

    if exp < 0 or exp > 65:
        messagebox.showerror("Error", "Please enter a value between 0 and 65.")
        return

    predicted_salary = regressor.predict([[exp]])[0]
    messagebox.showinfo("Prediction", f"Estimated Salary: ${predicted_salary:,.2f}")

btn = tk.Button(app, text="Predict Salary", command=show_prediction, bg="darkgreen", fg="white", width=18)
btn.pack(pady=18)

app.mainloop()
