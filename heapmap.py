import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data for the heatmap
data = [
    [0.8, 0.2, 0.4, 0.6],
    [0.3, 0.5, 0.7, 0.1],
    [0.9, 0.3, 0.6, 0.2],
    [0.2, 0.4, 0.8, 0.5]
]

# Define custom colors
colors = ["#FFA07A", "#87CEEB", "#ADFF2F", "#FFD700", "#FF69B4"]

# Create a mask array
mask = np.zeros_like(data)
mask[2, 1] = True  # Masking the value at row 2, column 1

# Create the heatmap using Seaborn with masking
sns.heatmap(data,
            annot=True,               # Add data annotations
            cmap=colors,              # Set custom colors
            linewidths=0.5,
            linecolor='gray',
            cbar=True,
            cbar_kws={"orientation": "vertical"},
            square=True,
            fmt='.2f',
            mask=mask,                # Apply the mask
            xticklabels=['A', 'B', 'C', 'D'],
            yticklabels=['W', 'X', 'Y', 'Z'])

# Set title and labels
plt.title('Sample Heatmap')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Display the heatmap
plt.show()
