import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkcalendar import Calendar, DateEntry
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import os
import threading
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import Callback
import requests
import csv
from PIL import Image, ImageTk
import seaborn as sns
import telegram
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Set style for dark theme
plt.style.use('dark_background')
sns.set_style("darkgrid")

class NeuralNetworkStockPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Stock Price Predictor")
        self.root.geometry("1400x900")
        self.root.configure(bg='#121212')
        
        # Initialize variables
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = None
        self.X_train, self.y_train = None, None
        self.training_history = None
        self.is_training = False
        self.training_thread = None
        self.telegram_bot = None
        self.telegram_chat_id = None
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        # Create menu
        self.create_menu()
        
        # Create main frames
        self.create_main_frames()
        
        # Setup layout
        self.setup_layout()
        
        # Load default data
        self.load_default_data()
        
    def configure_styles(self):
        # Configure styles for dark theme
        self.style.configure('TFrame', background='#121212')
        self.style.configure('TLabel', background='#121212', foreground='white')
        self.style.configure('TButton', background='#2c2c2c', foreground='white')
        self.style.configure('TEntry', fieldbackground='#2c2c2c', foreground='white')
        self.style.configure('TCombobox', fieldbackground='#2c2c2c', foreground='white')
        self.style.configure('TLabelframe', background='#121212', foreground='white')
        self.style.configure('TLabelframe.Label', background='#121212', foreground='white')
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg='#2c2c2c', fg='white')
        file_menu.add_command(label="Load CSV", command=self.load_csv)
        file_menu.add_command(label="Save Model", command=self.save_model)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0, bg='#2c2c2c', fg='white')
        view_menu.add_command(label="Show Data", command=self.show_data)
        view_menu.add_command(label="Reset View", command=self.reset_view)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0, bg='#2c2c2c', fg='white')
        tools_menu.add_command(label="Data Preprocessing", command=self.data_preprocessing)
        tools_menu.add_command(label="Model Architecture", command=self.model_architecture)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        # Export menu
        export_menu = tk.Menu(menubar, tearoff=0, bg='#2c2c2c', fg='white')
        export_menu.add_command(label="Export Prediction", command=self.export_prediction)
        export_menu.add_command(label="Export to Telegram", command=self.export_to_telegram)
        export_menu.add_command(label="Export Chart", command=self.export_chart)
        menubar.add_cascade(label="Export", menu=export_menu)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0, bg='#2c2c2c', fg='white')
        settings_menu.add_command(label="API Configuration", command=self.api_configuration)
        settings_menu.add_command(label="Appearance", command=self.appearance_settings)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        
        self.root.config(menu=menubar)
        
    def create_main_frames(self):
        # Left frame for controls
        self.left_frame = ttk.Frame(self.root, width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.left_frame.pack_propagate(False)
        
        # Right frame for charts
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.chart_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        self.prediction_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.chart_tab, text="Charts")
        self.notebook.add(self.stats_tab, text="Statistics")
        self.notebook.add(self.prediction_tab, text="Predictions")
        
    def setup_layout(self):
        # Setup left frame controls
        self.setup_data_section()
        self.setup_training_section()
        self.setup_model_section()
        self.setup_export_section()
        
        # Setup chart tab
        self.setup_chart_tab()
        
        # Setup stats tab
        self.setup_stats_tab()
        
        # Setup prediction tab
        self.setup_prediction_tab()
        
    def setup_data_section(self):
        data_frame = ttk.LabelFrame(self.left_frame, text="Data Configuration", padding=10)
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(data_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.file_path = tk.StringVar(value="Select CSV file")
        ttk.Entry(data_frame, textvariable=self.file_path, width=25).grid(row=0, column=1, pady=5, padx=5)
        ttk.Button(data_frame, text="Browse", command=self.load_csv).grid(row=0, column=2, pady=5)
        
        ttk.Label(data_frame, text="Training Start:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.train_start = DateEntry(data_frame, width=18, background='darkblue', 
                                    foreground='white', borderwidth=2, date_pattern='y-mm-dd')
        self.train_start.grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(data_frame, text="Training End:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.train_end = DateEntry(data_frame, width=18, background='darkblue', 
                                  foreground='white', borderwidth=2, date_pattern='y-mm-dd')
        self.train_end.grid(row=2, column=1, pady=5, padx=5)
        
        ttk.Button(data_frame, text="Load Data", command=self.prepare_data).grid(row=3, column=0, columnspan=3, pady=10)
        
    def setup_training_section(self):
        training_frame = ttk.LabelFrame(self.left_frame, text="Training Configuration", padding=10)
        training_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(training_frame, text="Lookback Days:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.lookback = tk.IntVar(value=60)
        ttk.Entry(training_frame, textvariable=self.lookback, width=10).grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(training_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.epochs = tk.IntVar(value=50)
        ttk.Entry(training_frame, textvariable=self.epochs, width=10).grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(training_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.batch_size = tk.IntVar(value=32)
        ttk.Entry(training_frame, textvariable=self.batch_size, width=10).grid(row=2, column=1, pady=5, padx=5)
        
        ttk.Label(training_frame, text="Test Size:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.test_size = tk.DoubleVar(value=0.2)
        ttk.Entry(training_frame, textvariable=self.test_size, width=10).grid(row=3, column=1, pady=5, padx=5)
        
        button_frame = ttk.Frame(training_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Start Training", command=self.start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Training", command=self.stop_training).pack(side=tk.LEFT, padx=5)
        
    def setup_model_section(self):
        model_frame = ttk.LabelFrame(self.left_frame, text="Model Configuration", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="Hidden Layers:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.hidden_layers = tk.IntVar(value=2)
        ttk.Entry(model_frame, textvariable=self.hidden_layers, width=10).grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(model_frame, text="Neurons per Layer:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.neurons = tk.IntVar(value=50)
        ttk.Entry(model_frame, textvariable=self.neurons, width=10).grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(model_frame, text="Dropout Rate:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.dropout = tk.DoubleVar(value=0.2)
        ttk.Entry(model_frame, textvariable=self.dropout, width=10).grid(row=2, column=1, pady=5, padx=5)
        
        ttk.Label(model_frame, text="Learning Rate:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.learning_rate = tk.DoubleVar(value=0.001)
        ttk.Entry(model_frame, textvariable=self.learning_rate, width=10).grid(row=3, column=1, pady=5, padx=5)
        
        ttk.Button(model_frame, text="Build Model", command=self.build_model).grid(row=4, column=0, columnspan=2, pady=10)
        
    def setup_export_section(self):
        export_frame = ttk.LabelFrame(self.left_frame, text="Export", padding=10)
        export_frame.pack(fill=tk.X)
        
        ttk.Button(export_frame, text="Save Model", command=self.save_model).pack(fill=tk.X, pady=5)
        ttk.Button(export_frame, text="Export Prediction", command=self.export_prediction).pack(fill=tk.X, pady=5)
        ttk.Button(export_frame, text="Export to Telegram", command=self.export_to_telegram).pack(fill=tk.X, pady=5)
        
    def setup_chart_tab(self):
        # Create figure for charts
        self.chart_fig = Figure(figsize=(10, 8), dpi=100, facecolor='#2c2c2c')
        self.chart_canvas = FigureCanvasTkAgg(self.chart_fig, self.chart_tab)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar_frame = ttk.Frame(self.chart_tab)
        self.toolbar_frame.pack(fill=tk.X)
        
        ttk.Button(self.toolbar_frame, text="Price Chart", command=self.plot_price_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar_frame, text="Training History", command=self.plot_training_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar_frame, text="Prediction", command=self.plot_prediction).pack(side=tk.LEFT, padx=5)
        
    def setup_stats_tab(self):
        # Create text widget for statistics
        self.stats_text = tk.Text(self.stats_tab, bg='#2c2c2c', fg='white', font=('Consolas', 10))
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.stats_tab, orient=tk.VERTICAL, command=self.stats_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        
    def setup_prediction_tab(self):
        # Frame for prediction controls
        pred_control_frame = ttk.Frame(self.prediction_tab)
        pred_control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(pred_control_frame, text="Prediction Days:").pack(side=tk.LEFT)
        self.pred_days = tk.IntVar(value=30)
        ttk.Entry(pred_control_frame, textvariable=self.pred_days, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(pred_control_frame, text="Generate Prediction", command=self.generate_prediction).pack(side=tk.LEFT, padx=5)
        
        # Frame for prediction chart
        self.pred_fig = Figure(figsize=(10, 6), dpi=100, facecolor='#2c2c2c')
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, self.prediction_tab)
        self.pred_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def load_default_data(self):
        # Check if default data exists in bin folder
        bin_folder = "bin"
        if not os.path.exists(bin_folder):
            os.makedirs(bin_folder)
            
        # Create sample data if it doesn't exist
        default_file = os.path.join(bin_folder, "sample_stock_data.csv")
        if not os.path.exists(default_file):
            # Generate sample data
            dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='D')
            prices = np.random.normal(100, 20, len(dates)).cumsum()
            prices = np.abs(prices)  # Ensure positive prices
            
            df = pd.DataFrame({'Date': dates, 'Price': prices})
            df.to_csv(default_file, index=False)
            
        self.file_path.set(default_file)
        self.load_csv_data(default_file)
        
    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if file_path:
            self.file_path.set(file_path)
            self.load_csv_data(file_path)
            
    def load_csv_data(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
            # Check if data has required columns
            if 'Date' not in self.data.columns or 'Price' not in self.data.columns:
                # Try to infer columns
                if len(self.data.columns) >= 2:
                    self.data.columns = ['Date', 'Price'] + list(self.data.columns[2:])
                else:
                    messagebox.showerror("Error", "CSV must have at least two columns: Date and Price")
                    return
                    
            # Convert date column to datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Set date range for training
            min_date = self.data['Date'].min()
            max_date = self.data['Date'].max()
            
            self.train_start.set_date(min_date)
            self.train_end.set_date(max_date - timedelta(days=30))  # Leave last 30 days for testing
            
            messagebox.showinfo("Success", f"Data loaded successfully!\nRows: {len(self.data)}")
            self.plot_price_data()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
            
    def prepare_data(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return
            
        try:
            # Filter data based on selected date range
            start_date = self.train_start.get_date()
            end_date = self.train_end.get_date()
            
            filtered_data = self.data[
                (self.data['Date'] >= pd.to_datetime(start_date)) & 
                (self.data['Date'] <= pd.to_datetime(end_date))
            ]
            
            if len(filtered_data) == 0:
                messagebox.showerror("Error", "No data in selected date range")
                return
                
            # Scale the data
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaled_data = self.scaler.fit_transform(filtered_data['Price'].values.reshape(-1, 1))
            
            # Create training data structure
            lookback = self.lookback.get()
            X, y = [], []
            
            for i in range(lookback, len(self.scaled_data)):
                X.append(self.scaled_data[i-lookback:i, 0])
                y.append(self.scaled_data[i, 0])
                
            X, y = np.array(X), np.array(y)
            
            # Reshape for LSTM [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Split into train and test sets
            test_size = int(len(X) * self.test_size.get())
            self.X_train, self.X_test = X[:-test_size], X[-test_size:]
            self.y_train, self.y_test = y[:-test_size], y[-test_size:]
            
            messagebox.showinfo("Success", 
                               f"Data prepared successfully!\nTraining samples: {len(self.X_train)}\nTest samples: {len(self.X_test)}")
                               
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare data: {str(e)}")
            
    def build_model(self):
        try:
            # Clear previous model
            if self.model is not None:
                del self.model
                
            # Build new model
            self.model = Sequential()
            
            # Input layer
            self.model.add(LSTM(
                units=self.neurons.get(),
                return_sequences=True,
                input_shape=(self.X_train.shape[1], 1)
            ))
            self.model.add(Dropout(self.dropout.get()))
            
            # Hidden layers
            for _ in range(self.hidden_layers.get() - 1):
                self.model.add(LSTM(units=self.neurons.get(), return_sequences=True))
                self.model.add(Dropout(self.dropout.get()))
                
            # Final LSTM layer
            self.model.add(LSTM(units=self.neurons.get()))
            self.model.add(Dropout(self.dropout.get()))
            
            # Output layer
            self.model.add(Dense(units=1))
            
            # Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate.get()),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            # Display model summary
            summary_list = []
            self.model.summary(print_fn=lambda x: summary_list.append(x))
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "\n".join(summary_list))
            
            messagebox.showinfo("Success", "Model built successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to build model: {str(e)}")
            
    def start_training(self):
        if self.model is None:
            messagebox.showerror("Error", "Please build the model first")
            return
            
        if self.X_train is None:
            messagebox.showerror("Error", "Please prepare data first")
            return
            
        # Disable training button and enable stop button
        self.is_training = True
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.train_model)
        self.training_thread.daemon = True
        self.training_thread.start()
        
    def stop_training(self):
        self.is_training = False
        messagebox.showinfo("Info", "Training will stop after current epoch")
        
    def train_model(self):
        try:
            # Custom callback to check for stop signal
            class TrainingCallback(Callback):
                def __init__(self, outer_instance):
                    self.outer_instance = outer_instance
                    
                def on_epoch_end(self, epoch, logs=None):
                    if not self.outer_instance.is_training:
                        self.model.stop_training = True
                        
            # Train the model
            self.training_history = self.model.fit(
                self.X_train, self.y_train,
                epochs=self.epochs.get(),
                batch_size=self.batch_size.get(),
                validation_data=(self.X_test, self.y_test),
                callbacks=[TrainingCallback(self)],
                verbose=0
            )
            
            # Evaluate the model
            train_loss, train_mae = self.model.evaluate(self.X_train, self.y_train, verbose=0)
            test_loss, test_mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            
            # Update stats tab
            self.root.after(0, self.update_training_stats, train_loss, train_mae, test_loss, test_mae)
            
            # Plot training history
            self.root.after(0, self.plot_training_history)
            
            if self.is_training:
                self.root.after(0, lambda: messagebox.showinfo("Success", "Training completed!"))
            else:
                self.root.after(0, lambda: messagebox.showinfo("Info", "Training stopped by user"))
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))
            
        finally:
            self.is_training = False
            
    def update_training_stats(self, train_loss, train_mae, test_loss, test_mae):
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "Training Results:\n")
        self.stats_text.insert(tk.END, f"Training Loss (MSE): {train_loss:.6f}\n")
        self.stats_text.insert(tk.END, f"Training MAE: {train_mae:.6f}\n")
        self.stats_text.insert(tk.END, f"Test Loss (MSE): {test_loss:.6f}\n")
        self.stats_text.insert(tk.END, f"Test MAE: {test_mae:.6f}\n\n")
        
        # Make predictions
        train_predict = self.model.predict(self.X_train)
        test_predict = self.model.predict(self.X_test)
        
        # Inverse transform predictions
        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict)
        
        # Inverse transform actual values
        y_train_actual = self.scaler.inverse_transform(self.y_train.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        # Calculate additional metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
        
        train_mape = np.mean(np.abs((y_train_actual - train_predict) / y_train_actual)) * 100
        test_mape = np.mean(np.abs((y_test_actual - test_predict) / y_test_actual)) * 100
        
        self.stats_text.insert(tk.END, f"Training RMSE: {train_rmse:.6f}\n")
        self.stats_text.insert(tk.END, f"Test RMSE: {test_rmse:.6f}\n")
        self.stats_text.insert(tk.END, f"Training MAPE: {train_mape:.2f}%\n")
        self.stats_text.insert(tk.END, f"Test MAPE: {test_mape:.2f}%\n")
        
    def generate_prediction(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first")
            return
            
        try:
            days = self.pred_days.get()
            
            # Use the last lookback days from the data to make prediction
            lookback = self.lookback.get()
            last_sequence = self.scaled_data[-lookback:]
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days):
                # Reshape for prediction
                X_pred = current_sequence.reshape(1, lookback, 1)
                
                # Predict next value
                pred = self.model.predict(X_pred, verbose=0)
                predictions.append(pred[0, 0])
                
                # Update sequence
                current_sequence = np.append(current_sequence[1:], pred)
                
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            # Generate future dates
            last_date = self.data['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
            
            # Plot predictions
            self.plot_future_predictions(future_dates, predictions)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            
    def plot_price_data(self):
        if self.data is None:
            return
            
        self.chart_fig.clear()
        ax = self.chart_fig.add_subplot(111)
        
        ax.plot(self.data['Date'], self.data['Price'], linewidth=1.5, color='cyan')
        ax.set_title('Stock Price History', color='white', fontsize=14)
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Price', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        self.chart_fig.autofmt_xdate()
        
        self.chart_canvas.draw()
        
    def plot_training_history(self):
        if self.training_history is None:
            return
            
        self.chart_fig.clear()
        ax = self.chart_fig.add_subplot(111)
        
        # Plot training & validation loss values
        ax.plot(self.training_history.history['loss'], label='Training Loss', color='cyan')
        ax.plot(self.training_history.history['val_loss'], label='Validation Loss', color='magenta')
        ax.set_title('Model Loss', color='white', fontsize=14)
        ax.set_ylabel('Loss', color='white')
        ax.set_xlabel('Epoch', color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        self.chart_canvas.draw()
        
    def plot_prediction(self):
        if self.model is None or self.X_test is None:
            return
            
        # Make predictions
        train_predict = self.model.predict(self.X_train)
        test_predict = self.model.predict(self.X_test)
        
        # Inverse transform predictions
        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict)
        
        # Prepare data for plotting
        lookback = self.lookback.get()
        
        # Create index for training predictions
        train_plot = np.empty_like(self.scaled_data)
        train_plot[:, :] = np.nan
        train_plot[lookback:lookback+len(train_predict), :] = train_predict
        
        # Create index for test predictions
        test_plot = np.empty_like(self.scaled_data)
        test_plot[:, :] = np.nan
        test_plot[lookback+len(train_predict):lookback+len(train_predict)+len(test_predict), :] = test_predict
        
        # Inverse transform actual data
        actual_data = self.scaler.inverse_transform(self.scaled_data)
        
        # Get dates for plotting
        start_date = self.train_start.get_date()
        end_date = self.train_end.get_date()
        filtered_data = self.data[
            (self.data['Date'] >= pd.to_datetime(start_date)) & 
            (self.data['Date'] <= pd.to_datetime(end_date))
        ]
        dates = filtered_data['Date'].values
        
        self.chart_fig.clear()
        ax = self.chart_fig.add_subplot(111)
        
        # Plot actual prices
        ax.plot(dates, actual_data, label='Actual Price', color='white', linewidth=1.5)
        
        # Plot training predictions
        ax.plot(dates, train_plot, label='Training Prediction', color='cyan', linewidth=1.5)
        
        # Plot test predictions
        ax.plot(dates, test_plot, label='Test Prediction', color='magenta', linewidth=1.5)
        
        ax.set_title('Model Predictions vs Actual Prices', color='white', fontsize=14)
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Price', color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        self.chart_fig.autofmt_xdate()
        
        self.chart_canvas.draw()
        
    def plot_future_predictions(self, future_dates, predictions):
        self.pred_fig.clear()
        ax = self.pred_fig.add_subplot(111)
        
        # Plot historical data
        historical_dates = self.data['Date'].values[-100:]  # Last 100 days
        historical_prices = self.data['Price'].values[-100:]
        
        ax.plot(historical_dates, historical_prices, label='Historical Price', color='cyan', linewidth=1.5)
        
        # Plot predictions
        ax.plot(future_dates, predictions, label='Predicted Price', color='magenta', linewidth=1.5)
        
        ax.set_title('Future Price Predictions', color='white', fontsize=14)
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Price', color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        self.pred_fig.autofmt_xdate()
        
        self.pred_canvas.draw()
        
    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No model to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".h5",
            filetypes=(("H5 files", "*.h5"), ("All files", "*.*"))
        )
        
        if file_path:
            try:
                self.model.save(file_path)
                messagebox.showinfo("Success", "Model saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
                
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=(("H5 files", "*.h5"), ("All files", "*.*"))
        )
        
        if file_path:
            try:
                self.model = load_model(file_path)
                messagebox.showinfo("Success", "Model loaded successfully!")
                
                # Display model summary
                summary_list = []
                self.model.summary(print_fn=lambda x: summary_list.append(x))
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, "\n".join(summary_list))
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                
    def export_prediction(self):
        if self.model is None:
            messagebox.showerror("Error", "No model available for prediction")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Predictions",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        
        if file_path:
            try:
                # Generate predictions
                days = self.pred_days.get()
                lookback = self.lookback.get()
                last_sequence = self.scaled_data[-lookback:]
                
                predictions = []
                current_sequence = last_sequence.copy()
                
                for _ in range(days):
                    X_pred = current_sequence.reshape(1, lookback, 1)
                    pred = self.model.predict(X_pred, verbose=0)
                    predictions.append(pred[0, 0])
                    current_sequence = np.append(current_sequence[1:], pred)
                
                # Inverse transform predictions
                predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                
                # Generate future dates
                last_date = self.data['Date'].iloc[-1]
                future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
                
                # Create DataFrame and save to CSV
                df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Price': predictions.flatten()
                })
                
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Predictions exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export predictions: {str(e)}")
                
    def export_to_telegram(self):
        # This is a placeholder for Telegram export functionality
        # In a real implementation, you would need to set up a Telegram bot
        # and obtain the necessary API credentials
        
        telegram_window = tk.Toplevel(self.root)
        telegram_window.title("Telegram Export Configuration")
        telegram_window.geometry("400x300")
        telegram_window.configure(bg='#121212')
        
        ttk.Label(telegram_window, text="Telegram Bot Token:").pack(pady=5)
        bot_token = ttk.Entry(telegram_window, width=40)
        bot_token.pack(pady=5)
        
        ttk.Label(telegram_window, text="Chat ID:").pack(pady=5)
        chat_id = ttk.Entry(telegram_window, width=40)
        chat_id.pack(pady=5)
        
        ttk.Label(telegram_window, text="Message:").pack(pady=5)
        message = tk.Text(telegram_window, height=5, width=40, bg='#2c2c2c', fg='white')
        message.pack(pady=5)
        
        def send_to_telegram():
            # This would be the actual implementation to send to Telegram
            messagebox.showinfo("Info", "Telegram export would be implemented here with proper API setup")
            
        ttk.Button(telegram_window, text="Send", command=send_to_telegram).pack(pady=10)
        
    def export_chart(self):
        file_path = filedialog.asksaveasfilename(
            title="Export Chart",
            defaultextension=".png",
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*"))
        )
        
        if file_path:
            try:
                self.chart_fig.savefig(file_path, dpi=300, facecolor='#2c2c2c', edgecolor='none')
                messagebox.showinfo("Success", f"Chart exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export chart: {str(e)}")
                
    def show_data(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to show")
            return
            
        data_window = tk.Toplevel(self.root)
        data_window.title("Stock Data")
        data_window.geometry("800x600")
        data_window.configure(bg='#121212')
        
        # Create treeview to display data
        columns = list(self.data.columns)
        tree = ttk.Treeview(data_window, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
            
        # Add scrollbar
        scrollbar = ttk.Scrollbar(data_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Add data to treeview
        for _, row in self.data.iterrows():
            tree.insert('', tk.END, values=list(row))
            
    def reset_view(self):
        self.notebook.select(0)  # Select chart tab
        self.plot_price_data()
        
    def data_preprocessing(self):
        preprocessing_window = tk.Toplevel(self.root)
        preprocessing_window.title("Data Preprocessing")
        preprocessing_window.geometry("400x300")
        preprocessing_window.configure(bg='#121212')
        
        ttk.Label(preprocessing_window, text="Data Preprocessing Options").pack(pady=10)
        
        # Add preprocessing options here
        ttk.Button(preprocessing_window, text="Normalize Data", command=self.normalize_data).pack(pady=5)
        ttk.Button(preprocessing_window, text="Remove Outliers", command=self.remove_outliers).pack(pady=5)
        ttk.Button(preprocessing_window, text="Fill Missing Values", command=self.fill_missing_values).pack(pady=5)
        
    def normalize_data(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to normalize")
            return
            
        # Simple normalization implementation
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        self.data['Price'] = scaler.fit_transform(self.data['Price'].values.reshape(-1, 1))
        
        messagebox.showinfo("Success", "Data normalized using StandardScaler")
        self.plot_price_data()
        
    def remove_outliers(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to process")
            return
            
        # Simple outlier removal using IQR
        Q1 = self.data['Price'].quantile(0.25)
        Q3 = self.data['Price'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        initial_count = len(self.data)
        self.data = self.data[
            (self.data['Price'] >= lower_bound) & 
            (self.data['Price'] <= upper_bound)
        ]
        removed_count = initial_count - len(self.data)
        
        messagebox.showinfo("Success", f"Removed {removed_count} outliers using IQR method")
        self.plot_price_data()
        
    def fill_missing_values(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to process")
            return
            
        # Check for missing values
        missing_count = self.data.isnull().sum().sum()
        
        if missing_count == 0:
            messagebox.showinfo("Info", "No missing values found")
            return
            
        # Fill missing values using interpolation
        self.data = self.data.interpolate()
        
        messagebox.showinfo("Success", f"Filled {missing_count} missing values using interpolation")
        self.plot_price_data()
        
    def model_architecture(self):
        architecture_window = tk.Toplevel(self.root)
        architecture_window.title("Model Architecture")
        architecture_window.geometry("500x400")
        architecture_window.configure(bg='#121212')
        
        ttk.Label(architecture_window, text="Model Architecture Visualization").pack(pady=10)
        
        # This would typically show a visualization of the model architecture
        # For simplicity, we'll just show the summary text
        
        summary_text = tk.Text(architecture_window, bg='#2c2c2c', fg='white')
        summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        if self.model is not None:
            summary_list = []
            self.model.summary(print_fn=lambda x: summary_list.append(x))
            summary_text.insert(tk.END, "\n".join(summary_list))
        else:
            summary_text.insert(tk.END, "No model built yet. Please build a model first.")
            
    def api_configuration(self):
        api_window = tk.Toplevel(self.root)
        api_window.title("API Configuration")
        api_window.geometry("400x300")
        api_window.configure(bg='#121212')
        
        ttk.Label(api_window, text="API Configuration").pack(pady=10)
        
        ttk.Label(api_window, text="OpenAI API Key:").pack(pady=5)
        openai_key = ttk.Entry(api_window, width=40)
        openai_key.pack(pady=5)
        
        ttk.Label(api_window, text="Alpha Vantage API Key:").pack(pady=5)
        av_key = ttk.Entry(api_window, width=40)
        av_key.pack(pady=5)
        
        def save_api_keys():
            # In a real implementation, you would save these securely
            messagebox.showinfo("Info", "API keys would be saved securely in a real implementation")
            
        ttk.Button(api_window, text="Save", command=save_api_keys).pack(pady=10)
        
    def appearance_settings(self):
        appearance_window = tk.Toplevel(self.root)
        appearance_window.title("Appearance Settings")
        appearance_window.geometry("300x200")
        appearance_window.configure(bg='#121212')
        
        ttk.Label(appearance_window, text="Appearance Settings").pack(pady=10)
        
        theme_var = tk.StringVar(value="Dark")
        ttk.Label(appearance_window, text="Theme:").pack(pady=5)
        ttk.Combobox(appearance_window, textvariable=theme_var, values=["Dark", "Light", "System"]).pack(pady=5)
        
        ttk.Label(appearance_window, text="Font Size:").pack(pady=5)
        font_size = ttk.Combobox(appearance_window, values=["10", "12", "14", "16"])
        font_size.set("12")
        font_size.pack(pady=5)
        
        def apply_settings():
            messagebox.showinfo("Info", "Appearance settings would be applied in a real implementation")
            
        ttk.Button(appearance_window, text="Apply", command=apply_settings).pack(pady=10)

def main():
    root = tk.Tk()
    app = NeuralNetworkStockPredictor(root)
    root.mainloop()

if __name__ == "__main__":
    main()