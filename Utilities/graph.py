import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load the data
data = pd.read_csv('/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/Data/n1cropped/center_n1cropped.csv')

# Create a figure and axis
fig, ax = plt.subplots()

# Initialize a scatter plot
scat = ax.scatter([], [], s=10)

# Set axis limits
ax.set_xlim(data['center_x'].min(), data['center_x'].max())
ax.set_ylim(data['center_y'].min(), data['center_y'].max())
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Fish Tracking')

# Update function for animation
def update(frame_number):
    # Filter data for the current frame
    frame_data = data[data['frame'] == frame_number]
    # Update scatter plot
    scat.set_offsets(frame_data[['center_x', 'center_y']].values)
    return scat,

# Create an animation
ani = animation.FuncAnimation(fig, update, frames=data['frame'].max(), interval=50, blit=True)

# Show the plot
plt.show()
