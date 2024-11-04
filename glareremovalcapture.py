from datetime import datetime
import serial
import numpy as np
from PIL import Image
from picamera2 import Picamera2
from libcamera import Transform
import time
from queue import Queue
import threading
import cv2
import os
from pyzbar.pyzbar import decode

class CameraSystem:
    def __init__(self):
        # Initialize the camera
        self.picam2 = Picamera2()

        # Configure the camera for 720p streaming suitable for OpenCV
        self.stream_config = self.picam2.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"},
            controls={"FrameRate": 30}
        )

        # Configure the camera for high-resolution capture
        self.capture_config = self.picam2.create_still_configuration(
            main={"size": (8000, 6000), "format": "RGB888"},
            transform=Transform()
        )

        self.picam2.configure(self.stream_config)
        self.picam2.start()

        # Lookup table for autofocus based on distance values
        self.lookup_table = [
            (596.39, 5.84), (553.72, 5.84), (662.57, 5.84), (597.82, 5.92),
            (530.6, 5.94), (512.97, 5.96), (476.2, 6.04), (454.77, 6.1),
            (434.14, 6.18), (420.9, 6.24), (405.72, 6.3), (386.71, 6.34),
            (381.2, 6.42), (364.7, 6.46), (353.45, 6.5), (351.52, 6.54),
            (350.78, 6.58)
        ]

        # Global variables
        self.latest_average_value = None
        self.message_queue = Queue()

    def interpolate_focus_position(self, average_value):
        lookup_table = sorted(self.lookup_table, key=lambda x: x[0])
        if average_value <= lookup_table[0][0]:
            return lookup_table[0][1]
        if average_value >= lookup_table[-1][0]:
            return lookup_table[-1][1]

        for i in range(len(lookup_table) - 1):
            if lookup_table[i][0] <= average_value <= lookup_table[i + 1][0]:
                avg1, pos1 = lookup_table[i]
                avg2, pos2 = lookup_table[i + 1]
                return pos1 + (pos2 - pos1) * (average_value - avg1) / (avg2 - avg1)
        return None

    def preview_thread(self):
        while True:
            try:
                frame = self.picam2.capture_array()
                cv2.imshow('Camera Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                # Handle exceptions (e.g., when camera is reconfigured)
                time.sleep(0.1)
        cv2.destroyAllWindows()

    def capture_and_process_images(self):
        print("multiplecapture")
        images = []
        image_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Stop the camera and preview
        self.picam2.stop()

        # Switch to capture configuration
        self.picam2.configure(self.capture_config)
        self.picam2.start()
        time.sleep(0.3)  # Allow time for the camera to adjust

        for i in range(3):
            frame_name = f"CAP_{timestamp}_frame{i+1}.jpg"
            self.picam2.capture_file(frame_name)
            images.append(Image.open(frame_name))
            image_paths.append(frame_name)
            print(f"Captured frame {i+1} as {frame_name}")
            time.sleep(0.5)  # Small delay between captures

        # Stop the camera after capture
        self.picam2.stop()

        # Switch back to preview configuration
        self.picam2.configure(self.stream_config)
        self.picam2.start()

        # Crop images to the specified size before processing
        # Calculate crop coordinates based on center and desired size
        center_x, center_y = 4000, 3000
        crop_size = 2000
        half_crop = crop_size // 2
        crop_left = center_x - half_crop
        crop_upper = center_y - half_crop
        crop_right = center_x + half_crop
        crop_lower = center_y + half_crop

        # Ensure crop coordinates are within image bounds
        image_width, image_height = images[0].size
        crop_left = max(0, crop_left)
        crop_upper = max(0, crop_upper)
        crop_right = min(image_width, crop_right)
        crop_lower = min(image_height, crop_lower)

        crop_box = (crop_left, crop_upper, crop_right, crop_lower)

        # Process the three images
        processed_image = self.process_images(image_paths, crop_box)
        
        # Barcode Detection
        start_time = time.time()
        barcodes = decode(processed_image)
        if not barcodes:
            print("No barcodes detected")
        else:
            print("Detected barcodes and QR codes")
            for barcode in barcodes:
                barcode_data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                print(f"{barcode_type}: {barcode_data}")
        end_time = time.time()
        
        print(f"barcode time:{end_time - start_time}")
                
        if processed_image is not None:
            # Save the processed image
            processed_image_filename = f"Processed_Image_{timestamp}.jpg"
            cv2.imwrite(processed_image_filename, processed_image)
            print(f"Processed image saved as {processed_image_filename}")
        else:
            print("Processing failed.")

    def capture_and_save_image(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("singlecapture")

        # Stop the camera and preview
        self.picam2.stop()

        # Switch to capture configuration
        self.picam2.configure(self.capture_config)
        self.picam2.start()
        time.sleep(0.3)  # Allow time for the camera to adjust

        # Capture one image
        frame_name = f"CAP_{timestamp}.jpg"
        self.picam2.capture_file(frame_name)
        image = Image.open(frame_name)
        print(f"Captured image as {frame_name}")

        # Stop the camera after capture
        self.picam2.stop()

        # Switch back to preview configuration
        self.picam2.configure(self.stream_config)
        self.picam2.start()

        # Calculate crop coordinates based on center and desired size
        center_x, center_y = 4000, 3000
        crop_size = 2000
        half_crop = crop_size // 2
        crop_left = center_x - half_crop
        crop_upper = center_y - half_crop
        crop_right = center_x + half_crop
        crop_lower = center_y + half_crop

        # Ensure crop coordinates are within image bounds
        image_width, image_height = image.size
        crop_left = max(0, crop_left)
        crop_upper = max(0, crop_upper)
        crop_right = min(image_width, crop_right)
        crop_lower = min(image_height, crop_lower)

        crop_box = (crop_left, crop_upper, crop_right, crop_lower)

        # Crop and save the image
        cropped_image = image.crop(crop_box)
        cropped_image_filename = f"Cropped_Image_{timestamp}.jpg"
        cropped_image.save(cropped_image_filename)
        print(f"Cropped image saved as {cropped_image_filename}")

    def serial_thread(self):
        try:
            ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            print("Serial port opened successfully.")
        except serial.SerialException as e:
            print(f"Failed to open serial port: {e}")
            return

        window_size = 10
        distance_values = []
        try:
            while True:
                if ser.in_waiting > 0:
                    data = ser.readline().decode('utf-8').strip()
                    try:
                        if data == "CAPTURE1":
                            self.message_queue.put("CAPTURE1")
                            continue
                        elif data == "CAPTURE":
                            self.message_queue.put("CAPTURE")
                            continue
                        distance_value = float(data)
                        distance_values.append(distance_value)
                        if len(distance_values) > window_size:
                            distance_values.pop(0)
                        self.latest_average_value = sum(distance_values) / len(distance_values)
                    except ValueError:
                        print("Invalid data received from serial.")
                        continue
        except KeyboardInterrupt:
            ser.close()

    def average_images(self, image_paths, crop_box):
        images = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Unable to read image {img_path}. Skipping this image.")
                continue
            # Crop the image
            x1, y1, x2, y2 = crop_box
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
            # Limit pixels with value 255 to 220
            cropped_img[cropped_img > 220] = 220
            # Convert to float32 for precise averaging
            img_float = cropped_img.astype(np.float32)
            images.append(img_float)

        if not images:
            print("Error: No images were loaded successfully.")
            return None

        # Compute the average
        averaged_image = np.mean(images, axis=0)
        # Clip values to [0, 255] and convert back to uint8
        averaged_image = np.clip(averaged_image, 0, 255).astype(np.uint8)
        return averaged_image

    def maximize_contrast(self, image):
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        # Merge the CLAHE enhanced L-channel with A and B channels
        limg = cv2.merge((cl, a, b))

        # Convert back to BGR color space
        contrast_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return contrast_enhanced

    def apply_clipping(self, image, clip_min, clip_max):
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Apply clipping on the V (brightness) channel
        v_clipped = np.clip(v, clip_min, clip_max).astype(np.uint8)

        # Merge back the channels
        hsv_clipped = cv2.merge([h, s, v_clipped])
        clipped_image = cv2.cvtColor(hsv_clipped, cv2.COLOR_HSV2BGR)

        return clipped_image

    def process_images(self, image_paths, crop_box):
        # Average the images
        averaged = self.average_images(image_paths, crop_box)
        if averaged is None:
            print("Error: Averaging failed.")
            return None

        # Enhance contrast
        contrast_enhanced = self.maximize_contrast(averaged)

        # Apply clipping
        clip_min = 0
        clip_max = 255
        processed_image = self.apply_clipping(contrast_enhanced, clip_min, clip_max)

        return processed_image

    def run(self):
        try:
            # Start serial reading thread
            serial_read_thread = threading.Thread(target=self.serial_thread)
            serial_read_thread.daemon = True
            serial_read_thread.start()

            # Start the preview thread
            preview_thread_instance = threading.Thread(target=self.preview_thread)
            preview_thread_instance.daemon = True
            preview_thread_instance.start()

            while True:
                # Set lens position based on latest TOF distance
                if self.latest_average_value is not None:
                    focus_position = self.interpolate_focus_position(self.latest_average_value)
                    if focus_position is not None:
                        self.picam2.set_controls({"LensPosition": focus_position})

                # Check message queue
                if not self.message_queue.empty():
                    message = self.message_queue.get()
                    if message == "CAPTURE1":
                        self.capture_and_save_image()
                    elif message == "CAPTURE":
                        self.capture_and_process_images()

                time.sleep(0.1)  # Delay to avoid high CPU usage

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.picam2.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_system = CameraSystem()
    camera_system.run()
