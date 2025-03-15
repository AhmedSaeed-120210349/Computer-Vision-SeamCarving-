import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_energy(image):
    """ Compute the energy of an image using the gradient magnitude """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(dx) + np.abs(dy)
    return energy



def find_seam(energy):
    """ Find the lowest-energy vertical seam """
    h, w = energy.shape
    cost = energy.copy()
    backtrack = np.zeros_like(cost, dtype=np.int64)


    for i in range(1, h):
        for j in range(w):
            left = cost[i-1, j-1] if j > 0 else float('inf')
            up = cost[i-1, j]
            right = cost[i-1, j+1] if j < w-1 else float('inf')
            
            min_index = np.argmin([left, up, right]) - 1     # -1 for left, 0 for up, 1 for right
            cost[i, j] += [left, up, right][min_index + 1]
            backtrack[i, j] = j + min_index

    seam = []
    j = np.argmin(cost[-1])
    for i in range(h-1, -1, -1):
        seam.append(j)
        j = backtrack[i, j]
    
    return seam[::-1]




def remove_seam(image, seam):
    """Remove a seam from an image."""
    h, w, _ = image.shape
    new_image = np.zeros((h, w-1, 3), dtype=np.uint8)
    
    for i in range(h):
        new_image[i, :, :] = np.delete(image[i, :, :], seam[i], axis=0)
    
    return new_image



def visualize_seams(image, seams):
    """Overlay seams on the image in red before removing them."""
    vis_image = image.copy()
    for seam in seams:
        for i in range(len(seam)):
            vis_image[i, seam[i]] = [0, 0, 255]  # Red color for seams
    return vis_image




def seam_carving(image_path, num_seams):
    """Perform seam carving on an image."""
    image = cv2.imread(image_path)
    seams = []

    for _ in range(num_seams):
        energy = compute_energy(image)
        seam = find_seam(energy)
        seams.append(seam)
        image = remove_seam(image, seam)
    
    vis_seams = visualize_seams(cv2.imread(image_path), seams)
    
    cv2.imwrite("resized.jpg", image)
    cv2.imwrite("seams_visualized.jpg", vis_seams)
    print("Processing complete. Output saved as 'resized.jpg' and 'seams_visualized.jpg'.")



# Run the algorithm
seam_carving(r"C:\Users\ahmed\OneDrive\Desktop\PIC.jpg", num_seams=100)   # Adjust the number of seams as needed
