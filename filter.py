import cv2 
import numpy as np
import matplotlib.pyplot as plt

kernels = {
    "Identity": np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]]),

    "Edge detection 1": np.array([[1, 0, -1],
                                   [0, 0,  0],
                                   [-1, 0, 1]]),

    "Edge detection 2": np.array([[0,  1, 0],
                                   [1, -4, 1],
                                   [0,  1, 0]]),

    "Edge detection 3": np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]]),

    "Sharpen": np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]]),

    "Box Blur": (1/9) * np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]]),

    "Gaussian Blur": (1/16) * np.array([[1, 2, 1],
                                        [2, 4, 2],
                                        [1, 2, 1]])
}


img = cv2.imread("camel_img.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = {}
for name, kernel in kernels.items():
    filtered = cv2.filter2D(img, -1, kernel)
    results[name] = filtered


plt.figure(figsize=(12, 8))
plt.subplot(2, 4, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

for i, (name, result) in enumerate(results.items(), start=2):
    plt.subplot(2, 4, i)
    plt.imshow(result)
    plt.title(name)
    plt.axis("off")
    

plt.tight_layout()
plt.show(block=True)