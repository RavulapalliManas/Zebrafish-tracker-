import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import ast
import csv

def plot_boxes(box_coordinates, image=None, figsize=(10, 8), color='r', linewidth=2, title=None):
    """
    Plot boxes on an image or on a blank canvas.
    
    Parameters:
    -----------
    box_coordinates : list or numpy array
        List of box coordinates in format [[x1, y1, x2, y2], ...] where
        (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    image : numpy array, optional
        Image on which to plot the boxes. If None, a blank canvas is used.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    color : str, optional
        Color of the box outlines.
    linewidth : int, optional
        Width of the box outlines.
    title : str, optional
        Title for the plot.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # If image is provided, display it
    if image is not None:
        ax.imshow(image)
    else:
        # Create a blank canvas with appropriate limits
        if len(box_coordinates) > 0:
            all_coords = np.array(box_coordinates)
            x_min, y_min = np.min(all_coords[:, [0, 1]], axis=0) - 10
            x_max, y_max = np.max(all_coords[:, [2, 3]], axis=0) + 10
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)  # Reversed for y-axis (origin at top-left)
        else:
            ax.set_xlim(0, 100)
            ax.set_ylim(100, 0)
    
    # Plot each box
    for box in box_coordinates:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=linewidth, 
                                edgecolor=color, facecolor='none')
        
        # Add the rectangle to the plot
        ax.add_patch(rect)
    
    if title:
        ax.set_title(title)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    return fig, ax

def show_boxes(box_coordinates, image=None, figsize=(10, 8), color='r', linewidth=2, title=None):
    """
    Plot and display boxes immediately.
    
    Parameters are the same as plot_boxes.
    """
    fig, ax = plot_boxes(box_coordinates, image, figsize, color, linewidth, title)
    plt.show()
    return fig, ax

def save_boxes_plot(box_coordinates, filename, image=None, figsize=(10, 8), 
                   color='r', linewidth=2, title=None, dpi=300):
    """
    Plot boxes and save the figure to a file.
    
    Parameters:
    -----------
    filename : str
        Path where to save the figure.
    dpi : int, optional
        Resolution of the saved figure.
    
    Other parameters are the same as plot_boxes.
    """
    fig, ax = plot_boxes(box_coordinates, image, figsize, color, linewidth, title)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return fig, ax

def plot_polygon_boxes(box_coordinates, image=None, figsize=(10, 8), color='r', linewidth=2, title=None):
    """
    Plot polygon boxes on an image or on a blank canvas.
    
    Parameters:
    -----------
    box_coordinates : list
        List of box coordinates in format [[(x1, y1), (x2, y2), (x3, y3), (x4, y4)], ...] 
        where each tuple represents a corner point.
    image : numpy array, optional
        Image on which to plot the boxes. If None, a blank canvas is used.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    color : str, optional
        Color of the box outlines.
    linewidth : int, optional
        Width of the box outlines.
    title : str, optional
        Title for the plot.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # If image is provided, display it
    if image is not None:
        ax.imshow(image)
    else:
        # Create a blank canvas with appropriate limits
        if len(box_coordinates) > 0:
            # Flatten all coordinates to find min and max
            all_points = []
            for box in box_coordinates:
                all_points.extend(box)
            
            all_points = np.array(all_points)
            x_min, y_min = np.min(all_points, axis=0) - 10
            x_max, y_max = np.max(all_points, axis=0) + 10
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)  # Reversed for y-axis (origin at top-left)
        else:
            ax.set_xlim(0, 100)
            ax.set_ylim(100, 0)
    
    # Plot each box
    for box in box_coordinates:
        # Convert points to arrays for plotting
        points = np.array(box)
        # Close the polygon by adding the first point at the end
        points = np.vstack([points, points[0]])
        
        # Plot the polygon
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth)
    
    if title:
        ax.set_title(title)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    return fig, ax

def show_polygon_boxes(box_coordinates, image=None, figsize=(10, 8), color='r', linewidth=2, title=None):
    """
    Plot and display polygon boxes immediately.
    
    Parameters are the same as plot_polygon_boxes.
    """
    fig, ax = plot_polygon_boxes(box_coordinates, image, figsize, color, linewidth, title)
    plt.show()
    return fig, ax

def load_boxes_from_csv(csv_file):
    """
    Load box coordinates from a CSV file in the format:
    Box 1,"[(129, 166), (433, 150), (446, 670), (141, 676)]"
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file.
        
    Returns:
    --------
    dict : Dictionary with box names as keys and coordinates as values
    """
    boxes = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                box_name = row[0]
                # Parse the string representation of coordinates
                try:
                    coords_str = row[1].strip()
                    coords = ast.literal_eval(coords_str)
                    boxes[box_name] = coords
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing coordinates for {box_name}: {e}")
    return boxes

def plot_boxes_from_csv(csv_file, image=None, figsize=(10, 8), color='r', linewidth=2, title=None):
    """
    Load boxes from a CSV file and plot them.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file.
    Other parameters are the same as plot_polygon_boxes.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    boxes_dict = load_boxes_from_csv(csv_file)
    box_coordinates = list(boxes_dict.values())
    
    if title is None:
        title = f"Boxes from {csv_file}"
    
    return plot_polygon_boxes(box_coordinates, image, figsize, color, linewidth, title)

def show_boxes_from_csv(csv_file, image=None, figsize=(10, 8), color='r', linewidth=2, title=None):
    """
    Load boxes from a CSV file and display them immediately.
    
    Parameters are the same as plot_boxes_from_csv.
    """
    fig, ax = plot_boxes_from_csv(csv_file, image, figsize, color, linewidth, title)
    plt.show()
    return fig, ax
