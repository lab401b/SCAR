import os
import urllib.request
import zipfile

def download_and_prepare_tiny_imagenet():
    # url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    url = "https://huggingface.co/datasets/terrys1095/test0410/resolve/main/tiny-imagenet-200.zip"

    data_dir = "./data"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    tiny_dir = os.path.join(data_dir, "tiny-imagenet-200")

    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(tiny_dir):
        print("⬇️ 正在下載 Tiny ImageNet (約 237MB)，請稍候...")
        urllib.request.urlretrieve(url, zip_path)
        print("📦 正在解壓縮...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

    # 重新整理 val 資料夾，讓 ImageFolder 可以正確讀取
    val_dir = os.path.join(tiny_dir, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')

    if os.path.exists(val_img_dir) and os.path.exists(val_annotations):
        print("🗂️ 正在重組驗證集資料夾結構...")
        with open(val_annotations, 'r') as f:
            for line in f.readlines():
                parts = line.split('\t')
                img_name, class_id = parts[0], parts[1]
                
                class_dir = os.path.join(val_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)
                
                src = os.path.join(val_img_dir, img_name)
                dst = os.path.join(class_dir, img_name)
                if os.path.exists(src):
                    os.rename(src, dst)
        os.rmdir(val_img_dir)
        print("✅ Tiny ImageNet 準備完畢！")
    else:
        print("✅ Tiny ImageNet 已經是最新狀態！")

if __name__ == "__main__":
    download_and_prepare_tiny_imagenet()