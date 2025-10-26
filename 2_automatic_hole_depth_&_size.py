# libraries required
# numpy, matplotlib, scipy, mpl_toolkits, json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import least_squares
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json




clicked_points = 'clicked_points_20250630_124212.json' 
path           = './files/' +  clicked_points
new_path       = './files/' +  'dict_'        +  clicked_points


# import json

def load_json(file_name):
    """
    Loads and returns data from a JSON file.

    Args:
        file_name (str): Path to the JSON file.

    Returns:
        dict or list: Parsed JSON content.
    """
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

# #example usage
# data = load_json('./files/clicked_points_20250324_112218.json')
# data



# import numpy as np

def find_lowest_point(npy_file, x, y, square_size):
    """
    Finds the lowest point in a square centered at (x, y) within a given .npy file.
    
    Parameters:
    - npy_file: str, path to the .npy file.
    - x: int, x-coordinate of the center.
    - y: int, y-coordinate of the center.
    - square_size: int, size of the square (e.g., 10, 20, 100).
    
    Returns:
    - (int, int): Indices of the most negative value within the square in original array.
    """
    # Load the array
    data = np.load(npy_file)                                              # Load the .npy file into a NumPy array
    
    # Get array shape
    height, width = data.shape                                            # Extract number of rows (height) and columns (width)
    
    # Define square boundaries ensuring they stay within the array limits
    half_size    = square_size // 2                                       # Half of the square size to determine boundaries
    x_min, x_max = max(0, x - half_size), min(width, x + half_size + 1)   # Ensure x boundaries stay within array limits
    y_min, y_max = max(0, y - half_size), min(height, y + half_size + 1)  # Ensure y boundaries stay within array limits
    
    # Extract the subarray
    subarray = data[y_min:y_max, x_min:x_max]                             # Slice out the relevant portion of the matrix
    
    # Find the indices of the smallest value within the subarray
    min_index = np.unravel_index(np.argmin(subarray), subarray.shape)     # Locate minimum value within the subarray
    
    # Convert local indices back to original array indices
    min_x = x_min + min_index[1]  # Convert local x index to global x index
    min_y = y_min + min_index[0]  # Convert local y index to global y index
    
    return min_x, min_y           # Return the global indices of the lowest point

# # Example usage (replace 'data.npy' with your actual .npy file path)
# npy_file     = 'scan_19_test_2.npy'                     # File path to .npy array
# x, y         = 130, 968                                 # Example coordinates representing the center of the search area
# square_size  = 20                                       # Define the size of the square to search within
# lowest_point = find_lowest_point(npy_file, x, y, 20)    # Call the function
# print("Lowest point indices:", lowest_point)            # Output the result












# import numpy as np

# def extract_annular_square_region(height_map, center_x, center_y, outer_length, inner_length):
#     if outer_length % 2 != 0 or inner_length % 2 != 0:
#         raise ValueError("Edge lengths must be even numbers.")
    
#     half_outer = outer_length // 2
#     half_inner = inner_length // 2
    
#     # Define the outer square corners
#     x_min_outer = max(0, center_x - half_outer)
#     x_max_outer = min(height_map.shape[1], center_x + half_outer)
#     y_min_outer = max(0, center_y - half_outer)
#     y_max_outer = min(height_map.shape[0], center_y + half_outer)
    
#     # Define the inner square corners
#     x_min_inner = max(0, center_x - half_inner)
#     x_max_inner = min(height_map.shape[1], center_x + half_inner)
#     y_min_inner = max(0, center_y - half_inner)
#     y_max_inner = min(height_map.shape[0], center_y + half_inner)
    
#     annular_coords = {
#         "top_strip": [],
#         "bottom_strip": [],
#         "left_strip": [],
#         "right_strip": []
#     }
#     heights = []
    
#     # Compute the annular width (thickness of the annular region)
#     annular_width = (outer_length - inner_length) // 2
    
#     # Extract bottom strip only if within bounds
#     if y_min_outer - annular_width >= 0:
#         for y in range(y_min_outer, min(y_min_outer + annular_width, height_map.shape[0])):
#             for x in range(x_min_outer, x_max_outer):
#                 annular_coords["bottom_strip"].append((x, y))
#                 heights.append(height_map[y, x])
    
#     # Extract top strip only if within bounds
#     if y_max_outer + annular_width < height_map.shape[0]:
#         for y in range(max(y_max_outer - annular_width, 0), y_max_outer):
#             for x in range(x_min_outer, x_max_outer):
#                 annular_coords["top_strip"].append((x, y))
#                 heights.append(height_map[y, x])
    
#     # Extract left strip only if it's within bounds
#     if x_min_outer - annular_width >= 0:
#         for y in range(y_min_inner, y_max_inner):
#             for x in range(x_min_outer, min(x_min_outer + annular_width, height_map.shape[1])):
#                 annular_coords["left_strip"].append((x, y))
#                 heights.append(height_map[y, x])
    
#     # Extract right strip only if it's within bounds
#     if x_max_inner + annular_width < height_map.shape[1]:
#         for y in range(y_min_inner, y_max_inner):
#             for x in range(x_max_inner, min(x_max_inner + annular_width, height_map.shape[1])):
#                 annular_coords["right_strip"].append((x, y))
#                 heights.append(height_map[y, x])
    
#     # Compute the average height value
#     avg_height = np.mean(heights) if heights else None
    
#     return avg_height, annular_coords, heights

# # #Example usage:
# # height_map = np.load("scan_19_test_2.npy")
# # avg, coords, heights = extract_annular_square_region(height_map, 29, 309, 102, 60)
# # print("Expected points count:", len(heights))











# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
# from scipy.optimize import least_squares

def fit_ellipse(x, y):
    """Fit an ellipse to a set of 2D points using least-squares optimization."""
    def ellipse_equation(params, x, y):
        """Equation of an ellipse: residuals for least-squares."""
        xc, yc, a, b, theta = params
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = cos_t * (x - xc) + sin_t * (y - yc)
        y_rot = -sin_t * (x - xc) + cos_t * (y - yc)
        return ((x_rot / a) ** 2 + (y_rot / b) ** 2 - 1)

    xc_guess = np.mean(x)
    yc_guess = np.mean(y)
    a_guess = (np.max(x) - np.min(x)) / 2
    b_guess = (np.max(y) - np.min(y)) / 2
    theta_guess = 0
    initial_params = [xc_guess, yc_guess, a_guess, b_guess, theta_guess]

    result = least_squares(ellipse_equation, initial_params, args=(x, y))
    xc, yc, a, b, theta = result.x
    return xc, yc, a, b, theta







# 

def plot_square_contour(filename, x_hole, y_hole, big_square_size):
    result = [None, None, None, None, None, None]  # [distance, major_axis, minor_axis, theta, all_levels, selected_level]

    data = np.load(filename)
    half_size = big_square_size // 2

    x_start = max(0, x_hole - half_size)
    x_end = min(data.shape[1], x_hole + half_size + 1)
    y_start = max(0, y_hole - half_size)
    y_end = min(data.shape[0], y_hole + half_size + 1)

    square_data = data[y_start:y_end, x_start:x_end]

    x = np.arange(x_start, x_end)
    y = np.arange(y_start, y_end)
    X, Y = np.meshgrid(x, y)

    # Plot 1: Contour with colorbar, aspect ratio preserved
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    contour = ax.contourf(X, Y, square_data, cmap='viridis', levels=30)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(contour, cax=cax, label='Height')

    all_levels_sorted = sorted(contour.levels)
    result[4] = [float(x) for x in all_levels_sorted]

    ax.set_title(f'Contour Map of the Hole')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
#    ax.grid(True, alpha=0.3)

    contour_lines = ax.contour(X, Y, square_data, levels=contour.levels, colors='white', alpha=0.5)

    def on_click(event):
        if result[0] is not None or event.inaxes != ax:
            return

        x_click, y_click = event.xdata, event.ydata
        min_dist = float('inf')
        selected_path = None
        selected_level = None

        for level_idx, level in enumerate(contour_lines.levels):
            segments = contour_lines.allsegs[level_idx]
            for segment in segments:
                for point in segment:
                    dist = np.sqrt((point[0] - x_click) ** 2 + (point[1] - y_click) ** 2)
                    if dist < min_dist and dist < 5:
                        min_dist = dist
                        selected_path = segment
                        selected_level = level

        result[5] = selected_level

        if selected_path is not None:
            x_contour, y_contour = selected_path.T
            xc, yc, a, b, theta = fit_ellipse(x_contour, y_contour)

            distance = np.sqrt((xc - x_hole)**2 + (yc - y_hole)**2)
            print(f"Distance between ellipse center ({xc}, {yc}) and hole center ({x_hole}, {y_hole}): {distance}")

            result[0] = distance
            result[1] = 2 * max(a, b)
            result[2] = 2 * min(a, b)
            result[3] = theta

            # Plot 2: Ellipse overlay plot
            fig_ellipse, ax_ellipse = plt.subplots(figsize=(6, 6))
            ax_ellipse.set_aspect('equal')

            contour2 = ax_ellipse.contourf(X, Y, square_data, cmap='viridis', levels=30)
            ax_ellipse.contour(X, Y, square_data, levels=[selected_level], colors='white')

            ellipse = Ellipse((xc, yc), 2 * a, 2 * b, angle=np.degrees(theta),
                              edgecolor='red', facecolor='none', linewidth=2)
            ax_ellipse.add_patch(ellipse)

            ax_ellipse.plot(xc, yc, 'ro', label='Ellipse Center', markersize=8)
            ax_ellipse.plot(x_hole, y_hole, 'bo', label='Hole Center', markersize=8)

            divider2 = make_axes_locatable(ax_ellipse)
            cax2 = divider2.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(contour2, cax=cax2, label='Height')

            ax_ellipse.set_title(f'Ellipse Fitted to Contour at height = {selected_level:.2e} m')
            ax_ellipse.set_xlabel('X')
            ax_ellipse.set_ylabel('Y')
#            ax_ellipse.grid(True, alpha=0.3)
            ax_ellipse.legend()

#            plt.savefig("contour_plot_ellipse.png", bbox_inches='tight', dpi=300)
            plt.show()

    fig.canvas.mpl_connect('button_press_event', on_click)
#    plt.savefig("contour_plot.png", bbox_inches='tight', dpi=1200)
    plt.show()

    return result[0], result[1], result[2], np.degrees(result[3]) if result[3] is not None else None, result[4], result[5]









    
def hole_depth_and_size_main(path_of_file_from_interactive_clicks):    
    file_path = path_of_file_from_interactive_clicks #"./files/clicked_points_20250404_130241.json" #clicked_points_20250328_104528.json"
    data = load_json(file_path)
    
    dict_of_scan_names_hole_quantities_holes_depths_and_average_depths = {}
    
    dims_of_scans = int(input("Kindly enter the size of all scans in pixels: "))
    scan_area     = int(input("Kindly enter scan area in square micrometers: "))
    for scan_name in data.keys():
        print("\033[1;34mSCAN_NAME: {scan_name}\033[0m".format(scan_name=scan_name))
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]                                   = {}
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["total_number_of_clicked_points"] = data[scan_name]["total_number_of_clicked_points"]
        
    
        
        npy_file_address = './files/'+scan_name                                                                # name of the file/scan that we are going to analyze
        npy_file         = np.load(npy_file_address)
    
        hole_number         = 0                                                   
        list_of_hole_depths                                = []
        list_of_distances_bw_hole_center_vs_ellipse_center = []
        list_of_major_axes_of_nanohole                     = []
        list_of_minor_axes_of_nanohole                     = []
        list_of_minor_over_major_axes                      = []
        list_of_angles_of_nanohole                         = []
        list_of_contour_heights_of_nanohole                = []




        
        for clicked_point in data[scan_name]['clicked_points']:
            print("clicked_point: ", clicked_point)
            hole_number                          +=  1                                                                 #----------------------------
            current_clicked_x, current_clicked_y  =   clicked_point[0], clicked_point[1]                               # Coordinates representing the center of the search area




            
    ########################################################################################
    # Depending on the typical sizes of holes you can adjust the square_size parameter to make the code more efficient 
    # For sample 22R033 square_size=20 is fine 
            if dims_of_scans    == 1024:
                square_size_for_lowest_point_finder                     =  20                                
                big_square_size_for_highest_point_finder                =  41
                small_square_size_for_highest_point_finder              =  5
                length_of_line_for_height_profile_extractors            =  41
                outer_square_length_for_annular_square_region_extractor =  102
                inner_square_length_for_annular_square_region_extractor =  60
                pixels_to_nano_meters                                   =  5000/dims_of_scans  #4.8828   # if scan size is not 5*5, u'll have to change this line accordingly.
     
            elif dims_of_scans  == 2048:
                square_size_for_lowest_point_finder           =  40
                big_square_size_for_highest_point_finder      =  81
                small_square_size_for_highest_point_finder    =  11
                length_of_line_for_height_profile_extractors  =  81
                outer_square_length_for_annular_square_region_extractor =  204
                inner_square_length_for_annular_square_region_extractor =  120
                pixels_to_nano_meters                                   =  5000/dims_of_scans  #2.4414
            
            else:
                print("Size of the scan is neither 1024px nor 2048px. Will use default values.")
                square_size_for_lowest_point_finder           =  40
                big_square_size_for_highest_point_finder      =  81
                small_square_size_for_highest_point_finder    =  11
                length_of_line_for_height_profile_extractors  =  81
                outer_square_length_for_annular_square_region_extractor =  204
                inner_square_length_for_annular_square_region_extractor =  120
                pixels_to_nano_meters                                   =  5000/dims_of_scans
            # square_size    =  40
    ########################################################################################





            
# # Extracting HOLE center | # Finding the coords of hole within the square around clicked point
            lowest_point =  find_lowest_point(
                                                npy_file_address,
                                                current_clicked_x,
                                                current_clicked_y,
                                                square_size_for_lowest_point_finder
                                                )
            current_hole_x, current_hole_y  =  int(lowest_point[0]), int(lowest_point[1])
            print("Coords of hole_number {hole_number} are: ({current_hole_x}, {current_hole_y})".format(hole_number=hole_number, current_hole_x=current_hole_x, current_hole_y=current_hole_y))    #---------------------------
    
    
    
            
    
    
# # Extracting Hole Shape
            x='y'
            while x == 'y':
                distance, major_axis, minor_axis, theta, all_levels, selected_level = plot_square_contour(
                    npy_file_address,
                    current_hole_x,
                    current_hole_y,
                    big_square_size_for_highest_point_finder
                )
                x = str(input('Do you want to repeat current iteration? (y/n): '))

            
            print('Distance (nm):', distance*pixels_to_nano_meters)
            print('Major axis (nm):', major_axis*pixels_to_nano_meters)
            print('Minor axis (nm):', minor_axis*pixels_to_nano_meters)
            print('Theta (deg):', theta)
            print('\n')
            print('All contour heights (nm) (sorted):', [x*pixels_to_nano_meters for x in all_levels])
            print('\n')
            print('Selected contour height (nm):', selected_level)

            
            list_of_distances_bw_hole_center_vs_ellipse_center.append(float(distance*pixels_to_nano_meters))
            list_of_major_axes_of_nanohole.append(float(major_axis*pixels_to_nano_meters))
            list_of_minor_axes_of_nanohole.append(float(minor_axis*pixels_to_nano_meters))
            list_of_minor_over_major_axes.append(float((minor_axis*pixels_to_nano_meters)/(major_axis*pixels_to_nano_meters)))
            list_of_angles_of_nanohole.append(float(theta))
            list_of_contour_heights_of_nanohole.append(all_levels)
            

            dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["hole_number_{hole_number}".format(hole_number=hole_number)] = {}
        
            dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["hole_number_{hole_number}".format(hole_number=hole_number)]['distance_h_e']            =  round(float(distance*pixels_to_nano_meters),4)     # distance_bw_hole_center_vs_ellipse_center
            dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["hole_number_{hole_number}".format(hole_number=hole_number)]['major_axis']              =  round(float(major_axis*pixels_to_nano_meters),4)
            dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["hole_number_{hole_number}".format(hole_number=hole_number)]['minor_axis']              =  round(float(minor_axis*pixels_to_nano_meters),4)
            dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["hole_number_{hole_number}".format(hole_number=hole_number)]['minor_over_major']        =  round(float((minor_axis*pixels_to_nano_meters)/(major_axis*pixels_to_nano_meters)),4)
            dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["hole_number_{hole_number}".format(hole_number=hole_number)]['theta']                   =  round(float(theta),4)
            dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["hole_number_{hole_number}".format(hole_number=hole_number)]['selected_contour_height'] =  float(selected_level)

            
            ##################################################################################################

            print("")


# # Extracting Nanohole depth

            depth_of_nanohole = selected_level - npy_file[current_hole_y, current_hole_x]  # IMPORTANT!!!
            print('depth_of_nanohole: ', depth_of_nanohole)
            
            dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["hole_number_{hole_number}".format(hole_number=hole_number)]['depth']                   = depth_of_nanohole
            list_of_hole_depths.append(float(depth_of_nanohole))

            dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["hole_number_{hole_number}".format(hole_number=hole_number)]['all_contour_heights']     =  all_levels
            dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["hole_number_{hole_number}".format(hole_number=hole_number)]['useable']                 =  'true'
            

            print('\n', scan_name, '\n')



            
#            print('\n avg_base_height_around_current_nanohole: ', selected_level)
            print('\n\n\n\n\n')


        # computing and adding standard deviations to the dictionary
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["depths_std"]           = float(np.std(list_of_hole_depths))
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["distances_std"]        = float(np.std(list_of_distances_bw_hole_center_vs_ellipse_center))
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["major_axes_std"]       = float(np.std(list_of_major_axes_of_nanohole))
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["minor_axes_std"]       = float(np.std(list_of_minor_axes_of_nanohole))
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["minor_over_major_std"] = float(np.std(list_of_minor_over_major_axes))
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["angles_std"]           = float(np.std(list_of_angles_of_nanohole))



        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["average_hole_depth_for_{scan_name}".format(scan_name=scan_name)] = float(np.average(list_of_hole_depths))
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["density_of_holes_for_{scan_name}".format(scan_name=scan_name)]   = dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["total_number_of_clicked_points"]/scan_area



        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["list_of_distances_bw_hole_center_vs_ellipse_center_for_{scan_name}".format(scan_name=scan_name)] = list_of_distances_bw_hole_center_vs_ellipse_center
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["list_of_major_axes_of_nanohole_for_{scan_name}".format(scan_name=scan_name)] =                     list_of_major_axes_of_nanohole
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["list_of_minor_axes_of_nanohole_for_{scan_name}".format(scan_name=scan_name)] =                     list_of_minor_axes_of_nanohole
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["list_of_minor_over_major_axes_for_{scan_name}".format(scan_name=scan_name)] =                      list_of_minor_over_major_axes
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["list_of_angles_of_nanohole_for_{scan_name}".format(scan_name=scan_name)] =                         list_of_angles_of_nanohole
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["list_of_contour_heights_of_nanohole_for_{scan_name}".format(scan_name=scan_name)] =                list_of_contour_heights_of_nanohole

        

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print("")

        
        with open(new_path, 'w') as file:
            json.dump(dict_of_scan_names_hole_quantities_holes_depths_and_average_depths, file)

        print(dict_of_scan_names_hole_quantities_holes_depths_and_average_depths)
    
    return dict_of_scan_names_hole_quantities_holes_depths_and_average_depths




dict_of_scan_names_hole_quantities_holes_depths_and_average_depths = hole_depth_and_size_main(path)  








