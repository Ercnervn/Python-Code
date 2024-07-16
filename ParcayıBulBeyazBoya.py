import gxipy as gx
import os
from datetime import datetime
from PIL import Image
import numpy as np
import cv2

# Fotoğrafların kaydedileceği ana dizin
output_dir = 'your/output/directory'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Kamera dizinlerini oluştur
camera_dirs = {1: os.path.join(output_dir, 'kamera1'), 2: os.path.join(output_dir, 'kamera2')}
for camera_dir in camera_dirs.values():
    if not os.path.exists(camera_dir):
        os.makedirs(camera_dir)

# Birleştirilmiş görsellerin kaydedileceği dizin
merged_output_dir = os.path.join(output_dir, 'merged')
if not os.path.exists(merged_output_dir):
    os.makedirs(merged_output_dir)

# Çıkartılmış alanın kaydedileceği dizin
extracted_output_dir = os.path.join(output_dir, 'extracted')
if not os.path.exists(extracted_output_dir):
    os.makedirs(extracted_output_dir)

def get_next_index(camera_dir):
    files = [f for f in os.listdir(camera_dir) if f.endswith('.jpg')]
    return len(files) + 1

def save_image(image, filename):
    image.save(filename)
    print(f"Resim kaydedildi: {filename}")

def capture_image(camera, index, camera_dir):
    raw_image = camera.data_stream[0].get_image()
    if raw_image is None:
        print(f"Kameradan görüntü alınamadı: {index}")
        return None
    
    rgb_image = raw_image.convert("RGB")
    if rgb_image is None:
        print(f"Görüntü dönüştürülemedi: {index}")
        return None
    
    numpy_image = rgb_image.get_numpy_array()
    pil_image = Image.fromarray(numpy_image)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(camera_dir, f"kamera_{index}_{timestamp}.jpg")
    
    save_image(pil_image, filename)
    return filename

def merge_images(image1_path, image2_path, output_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)
    
    new_image = Image.new('RGB', (total_width, max_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))
    
    new_image.save(output_path)
    print(f"Birleştirilmiş resim kaydedildi: {output_path}")

def detect_and_crop_object(image_path, cropped_output_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to find the dark areas
    _, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask for the largest contour
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Extract the object using the mask
        object_extracted = cv2.bitwise_and(image, image, mask=mask)
        
        # Create a white background image
        background = np.ones_like(image, dtype=np.uint8) * 255
        
        # Apply the mask to the white background
        background = cv2.bitwise_and(background, background, mask=mask)
        
        # Add the object to the white background
        final_image = cv2.add(background, object_extracted)
        
        # Save the cropped image
        cv2.imwrite(cropped_output_path, final_image)
        print(f"Cisim kırpıldı ve kaydedildi: {cropped_output_path}")

def main():
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        print("Cihaz bulunamadı")
        return
    
    if dev_num < 2:
        print("En az iki kamera gereklidir")
        return
    
    cam1 = None
    cam2 = None
    
    try:
        cam1 = device_manager.open_device_by_index(2)
    except Exception as e:
        print(f"Kamera 1 açılamadı: {e}")

    try:
        cam2 = device_manager.open_device_by_index(1)
    except Exception as e:
        print(f"Kamera 2 açılamadı: {e}")

    if not cam1 or not cam2:
        if cam1:
            cam1.close_device()
        if cam2:
            cam2.close_device()
        print("Gerekli kameralar açılamadı. Program sonlandırılıyor.")
        return
    
    cam1.stream_on()
    cam2.stream_on()

    next_index_cam1 = get_next_index(camera_dirs[1])
    next_index_cam2 = get_next_index(camera_dirs[2])

    image1_path = capture_image(cam1, next_index_cam1, camera_dirs[1])
    image2_path = capture_image(cam2, next_index_cam2, camera_dirs[2])
    
    cam1.stream_off()
    cam1.close_device()
    cam2.stream_off()
    cam2.close_device()

    if image1_path and image2_path:
        merged_image_path = os.path.join(merged_output_dir, f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        merge_images(image1_path, image2_path, merged_image_path)
        
        cropped_output_path = os.path.join(extracted_output_dir, f"cropped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        detect_and_crop_object(merged_image_path, cropped_output_path)

if __name__ == "__main__":
    main()
