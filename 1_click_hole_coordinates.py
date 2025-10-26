import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# Define the directory path
directory = "files"

# Ensure directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

# Function to process a file
def process_file(file_path, json_data, json_path):
    # Load the selected matrix
    matrix = np.load(file_path)
    
    # Get filename from path
    file_name = os.path.basename(file_path)
    
    # Initialize empty list for coordinates
    clicked_points = []

    # Function to save list to JSON file
    def save_to_file():
        json_data[file_name] = {
            "clicked_points": clicked_points,
            "total_number_of_clicked_points": len(clicked_points)
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)

    # Function to handle mouse click events
    def onclick(event):
        x = int(event.xdata)
        y = int(event.ydata)
        if x is not None and y is not None:
            clicked_points.append((x, y))
            save_to_file()
            print("Clicked points:", clicked_points)
            print("Number of points:", len(clicked_points))




    # fig, ax = plt.subplots(figsize=(10, 8), dpi=300)  
    # heatmap = ax.imshow(matrix, cmap='viridis')
    # plt.colorbar(heatmap)
    # ax.set_title(f'Interactive Height Map - {file_name} - Click to Select Points')
    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    # fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.savefig(os.path.join(directory, f"heatmap_{file_name}_highres.png"), dpi=1200, bbox_inches='tight')
    # plt.show()




    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Create heatmap
    heatmap = ax.imshow(matrix, cmap='viridis')
    
    # Add colorbar and labels
    plt.colorbar(heatmap)
    ax.set_title(f'Interactive Height Map - {file_name} - Click to Select Points')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    # Show plot and wait for user to close it
    plt.show()
    
    return clicked_points

# Generate a unique JSON filename based on current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., 20250313_143022
json_filename = f'clicked_points_{timestamp}.json'
json_path = os.path.join(directory, json_filename)

# Initialize JSON data as empty dictionary for this run
json_data = {}

# Main program loop
while True:
    # Get list of .npy files
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    print("Available .npy files in /files/:")
    for i, file in enumerate(npy_files, 1):
        print(f"{i}. {file}")

    # Ask user to select a file
    while True:
        file_choice = input("Enter the name of the file you want to open (including .npy): ")
        file_path = os.path.join(directory, file_choice)
        if os.path.exists(file_path):
            break
        print("File not found. Please try again.")

    # Process the selected file
    process_file(file_path, json_data, json_path)

    # After closing the plot, show scanned and not scanned files
    scanned_files = list(json_data.keys())
    not_scanned_files = [f for f in npy_files if f not in scanned_files]
    
    print("\nAlready scanned files:")
    if scanned_files:
        for i, file in enumerate(scanned_files, 1):
            print(f"{i}. {file}")
    else:
        print("None")
    
    print("\nFiles not yet scanned:")
    if not_scanned_files:
        for i, file in enumerate(not_scanned_files, 1):
            print(f"{i}. {file}")
    else:
        print("None")

    # Ask what to do next
    while True:
        choice = input("Do you want to (m)ove to another file or (e)xit? ").lower()
        if choice in ['m', 'e']:
            break
        print("Please enter 'm' or 'e'")

    if choice == 'e':
        print(f"Exiting program. Data saved to {json_filename}")
        break



