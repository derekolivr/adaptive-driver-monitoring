import cv2

# --- CONFIGURATION ---
VIDEO_PATH = "test/DashGaze_test.mp4"
# -------------------

# Global variables
ref_point = []
cropping = False
frame_copy = None

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # Draw a rectangle around the region of interest
        cv2.rectangle(frame_copy, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", frame_copy)

        # Print the coordinates
        x_start, y_start = ref_point[0]
        x_end, y_end = ref_point[1]
        width = x_end - x_start
        height = y_end - y_start
        
        print("\n--- Region Selected ---")
        print(f"Coordinates: (x={x_start}, y={y_start}, width={width}, height={height})")
        print("Copy these values into the preprocessing script.")
        print("-----------------------\n")


# Load the video and get the first frame
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video at {VIDEO_PATH}")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame of the video.")
    exit()

cap.release()

frame_copy = frame.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# Instructions
print("="*60)
print("INSTRUCTIONS:")
print("1. A window will appear with the first frame of your video.")
print("2. Click and drag your mouse to draw a rectangle over the DRIVER'S FACE view.")
print("3. The coordinates will be printed in the terminal. Copy them.")
print("4. Press 'r' to reset the selection and draw a new box (e.g., for the ROAD view).")
print("5. Once you have both sets of coordinates, press 'q' to quit.")
print("="*60)


while True:
    cv2.imshow("image", frame_copy)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"): # Reset the cropping region
        frame_copy = frame.copy()
        print("Selection reset. You can draw a new box.")

    elif key == ord("q"): # Quit
        break

cv2.destroyAllWindows()
