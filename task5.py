import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox


def ReadSignalFile(file_name):
    """
    Reads signal data from the specified file. Assumes the file format includes
    lines of index and value pairs, with the first two lines containing metadata
    to be ignored.
    """
    expected_indices = []
    expected_samples = []

    with open(file_name, 'r') as f:
        for _ in range(4):  # Skip first 4 lines (metadata)
            f.readline()

        # Read the signal data
        line = f.readline()
        while line:
            parts = line.strip().split()
            if len(parts) == 2:  # Ensuring it has both time and value
                V1 = int(parts[0])
                V2 = float(parts[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
            line = f.readline()

    return expected_indices, expected_samples


def moving_average(signal, window_size):
    """
    Computes the moving average of a given signal with a specified window size.
    """
    n = len(signal)
    averages = []

    # Compute the moving average using a sliding window
    for i in range(n - window_size + 1):
        window = signal[i:i + window_size]
        avg = sum(window) / window_size
        averages.append(avg)

    return averages


def compare_signals(original, computed, window_size):
    """
    Plots the original signal and computed moving average for comparison.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(original)), original, label='Original Signal', marker='o', color='blue')
    plt.plot(range(len(computed)), computed, label=f'Moving Average (Window Size {window_size})', marker='x', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Signal Value')
    plt.title(f'Comparison of Original Signal and Moving Average (Window Size {window_size})')
    plt.legend()
    plt.grid()
    plt.show()


def sharpening(signal):
    # Convert the signal to a numpy array
    x = np.array(signal)

    # First derivative: Y(n) = x(n) - x(n-1)
    first_derivative = x[1:] - x[:-1]
    first_derivative = np.append(first_derivative, 0)  # Padding with zero for consistency

    # Second derivative: Y(n) = x(n+1) - 2x(n) + x(n-1)
    second_derivative = x[2:] - 2 * x[1:-1] + x[:-2]
    second_derivative = np.append(second_derivative, 0)  # Padding with zero for consistency

    return first_derivative, second_derivative


def compare_results(calculated, expected, tolerance=1e-5):
    """Compares two arrays within a tolerance. Returns True if they match, else False."""
    return np.allclose(calculated, expected, atol=tolerance)


def convolution(signal1, signal2):
    length = len(signal1) + len(signal2) - 1
    new_signal = []
    for k in range(length):
        sum = 0
        for x in range(k, -1, -1):
            if len(signal1) <= len(signal2):
                if x >= len(signal1): continue
                if k - x >= len(signal2): continue
                sum += signal1[x] * signal2[k - x]
            else:
                if x >= len(signal2): continue
                if k - x >= len(signal1): continue
                sum += signal2[x] * signal1[k - x]

        new_signal.append(sum)

    return new_signal


# Define functions for button click events

def on_calculate_moving_avg():
    try:
        window_size = int(window_size_entry.get())
        if window_size <= 0:
            messagebox.showerror("Invalid Input", "Window size must be a positive integer.")
            return
        _, signal_data = ReadSignalFile('MovingAvg_input.txt')
        ma = moving_average(signal_data, window_size)
        print(f"Moving Average with Window Size {window_size}: {ma}")
        compare_signals(signal_data, ma, window_size)

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid integer for the window size.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def on_calculate_sharpening():
    try:
        _, signal_data = ReadSignalFile('Derivative_input.txt')
        first_derivative, second_derivative = sharpening(signal_data)
        print("First Derivative:", first_derivative)
        print("Second Derivative:", second_derivative)
        # You can also plot or compare results here as needed

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def on_calculate_convolution():
    try:
        _, signal1 = ReadSignalFile('Signal 1.txt')
        _, signal2 = ReadSignalFile('Signal 2.txt')
        convolved_signal = convolution(signal1, signal2)
        print("Convolved Signal:", convolved_signal)
        # Here you can also compare the convolution result with expected output

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Set up the GUI window
root = tk.Tk()
root.title("Signal Processing")

# Create a label and input field for the window size
window_size_label = tk.Label(root, text="Enter the window size for the moving average:")
window_size_label.pack(pady=10)

window_size_entry = tk.Entry(root)
window_size_entry.pack(pady=5)

# Create buttons for each action
moving_avg_button = tk.Button(root, text="Calculate Moving Average", command=on_calculate_moving_avg)
moving_avg_button.pack(pady=10)

sharpening_button = tk.Button(root, text="Perform Sharpening", command=on_calculate_sharpening)
sharpening_button.pack(pady=10)

convolution_button = tk.Button(root, text="Perform Convolution", command=on_calculate_convolution)
convolution_button.pack(pady=10)

# Start the GUI main loop
root.mainloop()
