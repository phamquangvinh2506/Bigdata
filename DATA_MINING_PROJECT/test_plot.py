import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Test basic plot
plt.figure(figsize=(8, 5))
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Test Plot')
plt.savefig('test_plot.png', dpi=100)
print('Plot saved to test_plot.png')
print(f'File exists: {os.path.exists("test_plot.png")}')
print(f'File size: {os.path.getsize("test_plot.png")} bytes')

# Test with seaborn
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.histplot([1,2,2,3,3,3,4,4,5], kde=True)
plt.title('Seaborn Test')
plt.savefig('test_seaborn.png', dpi=100)
print(f'Seaborn plot saved, size: {os.path.getsize("test_seaborn.png")} bytes')
