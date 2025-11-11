import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

data_frame = pd.read_csv("census.csv")
data_frame.columns = data_frame.columns.str.strip()

print("--- Data Frame Head ---")
print(data_frame.head())
print("\n")

print("--- Data Types ---")
print(data_frame.info())
print("\n")

num_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

ROWS = 2
COLS = 3
plt.figure(figsize=(18, 15))

print("--- Generating Numerical Data Visualizations (KDE) ---")

for i, col in enumerate(num_cols):
    if i >= ROWS * COLS:
        break
    plt.subplot(ROWS, COLS, i + 1)
    sns.kdeplot(data=data_frame, x=col, fill=True)
    plt.title(f"Distribution of {col} (KDE)")
    plt.xlabel(col)
    plt.ylabel("Density")

plt.subplots_adjust(hspace = 1.0, wspace = 0.4)
plt.show()

category_columns = data_frame.select_dtypes(include=['object']).columns

ROWS = 3
COLS = 3
plt.figure(figsize=(18, 15))

print("--- Generating Categorical Data Visualizations (Count Plots) ---")

for i, col in enumerate(category_columns):
    if i >= ROWS * COLS:
        break
    plt.subplot(ROWS, COLS, i + 1)
    sns.countplot(y=col, data=data_frame, palette='viridis')
    plt.title(f"Distribution of {col}")
    plt.xlabel("Count")
    plt.ylabel(col.replace('-', ' ').title())

plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
plt.show()

print("--- Generating Income vs Gender Histogram (Plotly) ---")

fig = px.histogram(
    data_frame, 
    x='income', 
    color='sex', 
    title='<b> Income Distribution by Gender', 
    barmode='group'
)

fig.update_layout(
    width=600, 
    height=500, 
    bargap=0.2,
    xaxis_title='Income Level',
    yaxis_title='Count of Individuals',
    legend_title='Gender'
)

fig.show()

print("Script execution complete. All visualizations should be displayed.")