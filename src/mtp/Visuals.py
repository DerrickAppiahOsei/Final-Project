import numpy as np
import matplotlib.pyplot as plt
import pyfiglet
# Load preprocessed data for a specific file (adjust file index as needed)
file_index = 7
imgs = np.load('/home/m3-learning/Documents/myML/Training/preprocessed_data/%03d_img.npz' % file_index, allow_pickle=True)
boxes = np.load('/home/m3-learning/Documents/myML/Training/preprocessed_data/%03d_box.npz' % file_index, allow_pickle=True)

# Initialize lists to store feature values for visualization
electron_hits_count = []
electron_hit_sizes = []
electron_hit_intensity = []
electron_hit_x_coords = []
electron_hit_y_coords = []

# Calculate features
for i in range(len(imgs.files)):
    img = imgs['arr_' + str(i)]
    box = boxes['arr_' + str(i)]
    
    # 1. Count of Electron Hits per Image
    electron_hits_count.append(len(box))
    
    # 2. Size/Area of Electron Hits
    sizes = [(b[2] - b[0]) * (b[3] - b[1]) for b in box]
    electron_hit_sizes.extend(sizes)
    
    # 3. Intensity of Electron Hits (mean pixel value within each box)
    intensities = [np.mean(img) for img in box]
    electron_hit_intensity.extend(intensities)
    
    # 4. Spatial distribution of Electron Hits (x, y coordinates)
    x_coords = [(b[1] + b[3]) / 2 for b in box]
    y_coords = [(b[0] + b[2]) / 2 for b in box]
    electron_hit_x_coords.extend(x_coords)
    electron_hit_y_coords.extend(y_coords)

    

# Visualization
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# 1. Count of Electron Hits per Image (Histogram)
axs[0, 0].hist(electron_hits_count, bins=20, color='skyblue', edgecolor='black')
axs[0, 0].set_title('Count of Electron Hits per Image')
axs[0, 0].set_xlabel('Count')
axs[0, 0].set_ylabel('Frequency')

# 2. Size/Area of Electron Hits (Histogram)
axs[0, 1].hist(electron_hit_sizes, bins=20, color='salmon', edgecolor='black')
axs[0, 1].set_title('Size/Area of Electron Hits')
axs[0, 1].set_xlabel('Size/Area')
axs[0, 1].set_ylabel('Frequency')

# 3. Distribution of Electron Hit Size (Boxplot)
axs[0, 2].boxplot(electron_hit_sizes, vert=False)
axs[0, 2].set_title('Distribution of Electron Hit Size')
axs[0, 2].set_xlabel('Size/Area')

# 4. Intensity of Electron Hits (Histogram)
axs[1, 0].hist(electron_hit_intensity, bins=20, color='lightgreen', edgecolor='black')
axs[1, 0].set_title('Intensity of Electron Hits')
axs[1, 0].set_xlabel('Intensity')
axs[1, 0].set_ylabel('Frequency')

# 5. Variability in Hits Counts across Images (Boxplot)
axs[1, 1].boxplot(electron_hits_count, vert=False)
axs[1, 1].set_title('Variability in Hits Counts across Images')
axs[1, 1].set_xlabel('Count')

# 6. Spatial Distribution of Electron Hits (Scatter plot)
axs[1, 2].scatter(electron_hit_x_coords, electron_hit_y_coords, color='orange', alpha=0.5)
axs[1, 2].set_title('Spatial Distribution of Electron Hits')
axs[1, 2].set_xlabel('X Coordinate')
axs[1, 2].set_ylabel('Y Coordinate')



plt.tight_layout()
plt.show()
