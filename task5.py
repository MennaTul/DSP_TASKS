import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox


# -------------- Read file ----------------

def ReadSignalFile(file_name):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        for _ in range(3):  # Skip first 3 lines (metadata)
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


# -------------- Testing ---------------

def Test(file_name, Your_indices, Your_samples):
    expectedIndices, expectedValues = ReadSignalFile(file_name)
    if ((len(Your_indices) != len(expectedIndices)) or (len(Your_samples) != len(expectedValues))):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expectedIndices[i]):
            print("Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(Your_samples)):
        if abs(Your_samples[i] - expectedValues[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print("Test case passed successfully")


# ------------ Validation ------------

def writeOut(file_name, sig, start=0):
    with open(file_name, 'w') as file:
        file.write(f"0\n")
        file.write(f"0\n")
        file.write(f"{len(sig)}\n")

        for index in range(start, len(sig) - abs(start)):
            if isinstance(sig[index], int):  # Check if the element is an integer
                file.write(f"{index} {int(sig[index + abs(start)])}\n")
            else:  # Otherwise, treat it as a float
                file.write(f"{index} {round(float(sig[index + abs(start)]), 3)}\n")


def validate_moving_avg(size):
    indices, samples = ReadSignalFile('avg_output.txt')
    if size == 3:
        Test('MovingAvg_out1.txt', indices, samples)
    elif size == 5:
        Test('MovingAvg_out2.txt', indices, samples)


# ------------ FUNCTIONS ----------

def moving_average(signal, window_size):
    n = len(signal)
    averages = []

    # Compute the moving average using a sliding window
    for i in range(n - window_size + 1):
        window = signal[i:i + window_size]
        avg = sum(window) / window_size
        averages.append(avg)

    writeOut('avg_output.txt', averages)
    validate_moving_avg(window_size)
    return averages


def compare_signals(original, computed, window_size):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(original)), original, label='Original Signal', marker='o', color='blue')
    plt.plot(range(len(computed)), computed, label=f'Moving Average (Window Size {window_size})', marker='x',
             color='red')
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
    # first_derivative = np.append(first_derivative, 0)  # Padding with zero for consistency

    # Second derivative: Y(n) = x(n+1) - 2x(n) + x(n-1)
    second_derivative = x[2:] - 2 * x[1:-1] + x[:-2]
    # second_derivative = np.append(second_derivative, 0)  # Padding with zero for consistency

    writeOut('d1Out.txt', first_derivative)
    writeOut('d2Out.txt', second_derivative)
    indices1, samples1 = ReadSignalFile('d1Out.txt')
    Test('1st_derivative_out.txt', indices1, samples1)
    indices2, samples2 = ReadSignalFile('d2Out.txt')
    Test('2nd_derivative_out.txt', indices2, samples2)

    return first_derivative, second_derivative


def compare_results(calculated, expected, tolerance=1e-5):
    return np.allclose(calculated, expected, atol=tolerance)


def convolution(signal1, signal2, start):
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

        new_signal.append(int(sum))

    writeOut('convOut.txt', new_signal, start)
    indices, samples = ReadSignalFile('convOut.txt')
    Test('Conv_output.txt', indices, samples)

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
        ind1, signal1 = ReadSignalFile('Signal 1.txt')
        ind2, signal2 = ReadSignalFile('Signal 2.txt')
        start = int(min(ind1) + min(ind2))
        convolved_signal = convolution(signal1, signal2, start)
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
