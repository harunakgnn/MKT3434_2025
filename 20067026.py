
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss, hinge_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

root = tk.Tk()
root.title("ML GUI - Homework 1 - Full Version ")
root.geometry("1000x600")

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

regression_tab = ttk.Frame(notebook)
classification_tab = ttk.Frame(notebook)
notebook.add(regression_tab, text="Regression")
notebook.add(classification_tab, text="Classification")

df = None

def load_dataset():
    global df
    file_path = filedialog.askopenfilename()
    if file_path:
        df = pd.read_csv(file_path)
        messagebox.showinfo("Dataset Loaded", f"Dataset loaded with shape: {df.shape}")

def handle_missing_data(method):
    global df
    if df is None:
        messagebox.showerror("Error", "Load dataset first.")
        return

    if method == "Mean Imputation":
        imputer = SimpleImputer(strategy="mean")
        df.iloc[:, :] = imputer.fit_transform(df)
    elif method == "Interpolation":
        df.interpolate(inplace=True)
    elif method == "Forward Fill":
        df.fillna(method="ffill", inplace=True)
    elif method == "Backward Fill":
        df.fillna(method="bfill", inplace=True)

    messagebox.showinfo("Missing Data", f"Missing data handled with: {method}")

def train_linear_regression(loss):
    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = HuberRegressor() if loss == "Huber" else LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if loss == "MSE":
            score = mean_squared_error(y_test, preds)
        elif loss == "MAE" or loss == "Huber":
            score = mean_absolute_error(y_test, preds)

        result_label.config(text=f"{loss} Score: {round(score, 4)}")
    except Exception as e:
        messagebox.showerror("Training Error", str(e))

def train_svr(kernel, C_val, epsilon_val, loss):
    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = SVR(kernel=kernel, C=float(C_val), epsilon=float(epsilon_val))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if loss == "MSE":
            score = mean_squared_error(y_test, preds)
        elif loss == "MAE" or loss == "Huber":
            score = mean_absolute_error(y_test, preds)

        result_label.config(text=f"SVR ({kernel}) - {loss}: {round(score, 4)}")
    except Exception as e:
        messagebox.showerror("SVR Error", str(e))

def train_logistic_regression(loss):
    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        if loss == "Cross-Entropy":
            score = log_loss(y_test, probs)
        elif loss == "Hinge":
            preds_bin = np.where(preds == y_test, 1, -1)
            y_bin = np.where(y_test == preds, 1, -1)
            score = hinge_loss(y_bin, preds_bin)

        acc = accuracy_score(y_test, preds)
        result_label_cls.config(text=f"{loss} Loss: {round(score, 4)} | Accuracy: {round(acc*100, 2)}%")
    except Exception as e:
        messagebox.showerror("Training Error", str(e))

def train_svm_classifier(kernel, C_val):
    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = SVC(kernel=kernel, C=float(C_val))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        result_label_cls.config(text=f"SVM ({kernel}) - Accuracy: {round(acc*100, 2)}%")
    except Exception as e:
        messagebox.showerror("SVM Error", str(e))

top_frame = ttk.Frame(regression_tab)
top_frame.pack(pady=10)

ttk.Button(top_frame, text="Load Dataset", command=load_dataset).grid(row=0, column=0, padx=5)
missing_options = ["Mean Imputation", "Interpolation", "Forward Fill", "Backward Fill"]
missing_var = tk.StringVar()
ttk.OptionMenu(top_frame, missing_var, missing_options[0], *missing_options, command=handle_missing_data).grid(row=0, column=1, padx=5)
loss_options = ["MSE", "MAE", "Huber"]
loss_var = tk.StringVar()
ttk.OptionMenu(top_frame, loss_var, loss_options[0], *loss_options).grid(row=0, column=2, padx=5)
ttk.Button(top_frame, text="Train Linear Regression", command=lambda: train_linear_regression(loss_var.get())).grid(row=0, column=3, padx=5)

kernel_var = tk.StringVar(value="rbf")
C_entry = tk.Entry(top_frame)
C_entry.insert(0, "1.0")
epsilon_entry = tk.Entry(top_frame)
epsilon_entry.insert(0, "0.1")

ttk.Label(top_frame, text="SVR Kernel").grid(row=1, column=0, padx=5)
ttk.OptionMenu(top_frame, kernel_var, "rbf", "linear", "poly", "rbf").grid(row=1, column=1, padx=5)
ttk.Label(top_frame, text="C").grid(row=1, column=2)
C_entry.grid(row=1, column=3)
ttk.Label(top_frame, text="Epsilon").grid(row=1, column=4)
epsilon_entry.grid(row=1, column=5)
ttk.Button(top_frame, text="Train SVR", command=lambda: train_svr(kernel_var.get(), C_entry.get(), epsilon_entry.get(), loss_var.get())).grid(row=1, column=6, padx=5)

result_label = ttk.Label(regression_tab, text="Result will appear here.", font=("Arial", 12))
result_label.pack(pady=20)

top_frame_cls = ttk.Frame(classification_tab)
top_frame_cls.pack(pady=10)

ttk.Button(top_frame_cls, text="Load Dataset", command=load_dataset).grid(row=0, column=0, padx=5)
missing_var_cls = tk.StringVar()
ttk.OptionMenu(top_frame_cls, missing_var_cls, missing_options[0], *missing_options, command=handle_missing_data).grid(row=0, column=1, padx=5)
loss_cls_options = ["Cross-Entropy", "Hinge"]
loss_var_cls = tk.StringVar()
ttk.OptionMenu(top_frame_cls, loss_var_cls, loss_cls_options[0], *loss_cls_options).grid(row=0, column=2, padx=5)
ttk.Button(top_frame_cls, text="Train Logistic Regression", command=lambda: train_logistic_regression(loss_var_cls.get())).grid(row=0, column=3, padx=5)

kernel_cls_var = tk.StringVar(value="rbf")
C_cls_entry = tk.Entry(top_frame_cls)
C_cls_entry.insert(0, "1.0")
ttk.Label(top_frame_cls, text="SVM Kernel").grid(row=1, column=0, padx=5)
ttk.OptionMenu(top_frame_cls, kernel_cls_var, "rbf", "linear", "poly", "rbf").grid(row=1, column=1, padx=5)
ttk.Label(top_frame_cls, text="C").grid(row=1, column=2)
C_cls_entry.grid(row=1, column=3)
ttk.Button(top_frame_cls, text="Train SVM Classifier", command=lambda: train_svm_classifier(kernel_cls_var.get(), C_cls_entry.get())).grid(row=1, column=4, padx=5)

result_label_cls = ttk.Label(classification_tab, text="Classification result will appear here.", font=("Arial", 12))
result_label_cls.pack(pady=20)

root.mainloop()
