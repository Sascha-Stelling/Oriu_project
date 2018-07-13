import cv2

def compute_optical_flow(img_prev, img_curr):
    # Computes the optical flow between two consecutive images img_prev at time step t and img_curr at time step t+1
    # Input: img_prev, img_curr (image_height x image_width); grayscale images at time step t and t+1
    # Output: flow: (image_height x image_width x 2); holds the optical flow vectors for the x and y direction
    #         mag: (image_height x image_width); magnitudes of the flow vectors
    #         ang: (image_height x image_width); orientation of the flow vectors
    flow = cv2.calcOpticalFlowFarneback(img_prev, img_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    return flow, mag, ang