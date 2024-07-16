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
camera_dirs = {
    1: os.path.join(output_dir, 'kamera1'),
    2: os.path.join(output_dir, 'kamera2')
}

for camera_dir in camera_dirs.values():
    if not os.path.exists(camera_dir):
        os.makedirs(camera_dir)

# Birleştirilmiş görsellerin kaydedileceği dizin
merged_output_dir = os.path.join(output_dir, 'merged')
if not os.path.exists(merged_output_dir):
    os.makedirs(merged_output_dir)

# Kenar tespit edilen görsellerin kaydedileceği dizin
edges_output_dir = os.path.join(output_dir, 'edges')
if not os.path.exists(edges_output_dir):
    os.makedirs(edges_output_dir)

def get_next_index(camera_dir):
    """Output directory'deki mevcut fotoğrafların sayısına göre bir sonraki indeks numarasını döner."""
    files = [f for f in os.listdir(camera_dir) if f.endswith('.jpg')]
    return len(files) + 1

def save_image(image, filename):
    # Görüntüyü kaydet
    image.save(filename)
    print(f"Resim kaydedildi: {filename}")

def capture_image(camera, index, camera_dir):
    # Kameradan fotoğraf çek
    raw_image = camera.data_stream[0].get_image()
    if raw_image is None:
        print(f"Kameradan görüntü alınamadı: {index}")
        return None
    
    # Görüntüyü işleyin
    rgb_image = raw_image.convert("RGB")
    if rgb_image is None:
        print(f"Görüntü dönüştürülemedi: {index}")
        return None
    
    # RGB görüntüyü Pillow Image nesnesine dönüştür
    numpy_image = rgb_image.get_numpy_array()
    pil_image = Image.fromarray(numpy_image)
    
    # Dosya adı oluştur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(camera_dir, f"kamera_{index}_{timestamp}.jpg")
    
    # Görüntüyü kaydet
    save_image(pil_image, filename)
    return filename

def merge_images(image1_path, image2_path, output_path):
    # Görselleri aç
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    
    # Görselleri yatay olarak birleştir
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)
    
    new_image = Image.new('RGB', (total_width, max_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))
    
    # Birleştirilmiş görüntüyü kaydet
    new_image.save(output_path)
    print(f"Birleştirilmiş resim kaydedildi: {output_path}")

def detect_edges(image_path, output_path):
    # Görseli aç ve gri tonlamalıya çevir
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Kenar tespiti yap
    edges = cv2.Canny(gray_image, 100, 200)
    
    # Kenar tespit edilen görseli kaydet
    cv2.imwrite(output_path, edges)
    print(f"Kenar tespiti yapılmış resim kaydedildi: {output_path}")

def main():
    # Kamera sistemini başlat
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        print("Cihaz bulunamadı")
        return
    
    # İki kamera kontrol edin
    if dev_num < 2:
        print("En az iki kamera gereklidir")
        return
    
    # İki kamerayı başlatın
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

    # Eğer kameralar açılamadıysa, programı sonlandırın
    if not cam1 or not cam2:
        if cam1:
            cam1.close_device()
        if cam2:
            cam2.close_device()
        print("Gerekli kameralar açılamadı. Program sonlandırılıyor.")
        return
    
    # Kameralardan görüntü alın
    cam1.stream_on()
    cam2.stream_on()

    next_index_cam1 = get_next_index(camera_dirs[1])
    next_index_cam2 = get_next_index(camera_dirs[2])

    image1_path = capture_image(cam1, next_index_cam1, camera_dirs[1])
    image2_path = capture_image(cam2, next_index_cam2, camera_dirs[2])
    
    # Kameraların veri akışını durdurun ve kapatın
    cam1.stream_off()
    cam1.close_device()
    
    cam2.stream_off()
    cam2.close_device()

    if image1_path and image2_path:
        # Yeni kaydedilen resim dosyalarını birleştir
        merged_image_path = os.path.join(merged_output_dir, f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        merge_images(image1_path, image2_path, merged_image_path)
        
        # Birleştirilmiş görsel üzerinde kenar tespiti yap
        edges_image_path = os.path.join(edges_output_dir, f"edges_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        detect_edges(merged_image_path, edges_image_path)

if __name__ == "__main__":
    main()
