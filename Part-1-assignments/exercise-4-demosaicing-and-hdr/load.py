import numpy as np
import matplotlib.pyplot as plt

# Load the raw image data
array = np.load('IMG_9939.npy')
print('Loaded array of size', array.shape) # (4014, 6020)

patch = array[1000:1016, 3000:3016]   

plt.imshow(patch, cmap='viridis')  
plt.title('A patch from image')
plt.colorbar()
plt.show()

print(patch[:4, :4])

# GRBG Bayer pattern, by analyzing above 4×4 block from the raw data
# (even, even) - Green
# (even, odd) - Red
# (odd, even) - Blue
# (odd, odd) - Green
