import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as nn_f
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from collections import Counter

# 對每個 Bad-PFL 模型逐圈刪除 FFT 頻率，找出刪掉哪個 radius 的頻率會讓 ASR 掉最多，把那個 radius 當成 trigger 的主要頻率位置，最後提供 Fig.3 的原始統計資料。

# ==========================================================
# ⚠️ 請確保您可以從您的專案中引入以下模組
from resnet import get_resnet 
from generator import Autoencoder
# ==========================================================

# ========================= ⚙️ 設定區 ==========================
# 📂 1. 模型集合主資料夾 (請改成您要測試的大資料夾)
# MODELS_BASE_DIR = "parameterChange0317/fedavg1000pC_0317/experiment_results_collection_fedavg_05" 01 to 05
MODELS_BASE_DIR = "experiment_results_collection_fedbn_noShuffle0303/experiment_results_collection_10" #記得改MODEL_FILENAME! from 1 to 10

# 📂 2. 資料夾內預設的檔名 (請根據情況修改，例如 fedbn 可能是 clean_client_87.pth)
# MODEL_FILENAME = "final_global_model.pth" 
MODEL_FILENAME = "clean_client_87.pth"
GENERATOR_FILENAME = "trained_trigger_generator.pth"

# 🎯 3. 測試設定
TRUE_LABEL = 8           # 原本乾淨圖片的正確標籤 
ATTACK_TARGET = 0        # 💀 Trigger 想讓模型誤判成的標籤
NUM_TEST_IMAGES = 1000   # 測試圖片數量 (若 GPU Out of Memory, 請調小至 500 或 200)

# 🎯 4. 掃描設定 (Bandstop)
R_MIN = 1                
R_MAX = 23               

# 🛡️ 5. 統計過濾設定
MIN_ASR_DROP_THRESHOLD = 0.20  # 🌟 新增：最大 ASR 跌幅必須大於 10% 才記錄              

# PGD 參數
PGD_EPSILON = 4./255.
PGD_ALPHA = 4./255.
PGD_ITER = 1

# 輸出設定
OUTPUT_CSV = "trigger_location_analysis_0329_15.csv"
OUTPUT_PLOT = "trigger_distribution_0329_15.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================================

def apply_bandstop_filter(images, radius):
    """
    專為此腳本簡化的 Bandstop 濾波器
    精準挖掉 (radius - 1) 到 radius 之間的頻率環
    """
    imgs_np = images.cpu().numpy()
    filtered_imgs = []
    rows, cols = imgs_np.shape[2], imgs_np.shape[3]
    crow, ccol = rows//2 , cols//2
    y, x = np.ogrid[:rows, :cols]
    dist_sq = (x - ccol)**2 + (y - crow)**2
    
    # 預設全過 (1)，只挖掉黑環 (0)
    mask = np.ones((rows, cols), dtype=np.float32)
    inner_radius = radius - 1
    mask[(dist_sq > inner_radius**2) & (dist_sq <= radius**2)] = 0
            
    for i in range(imgs_np.shape[0]):
        img_channels = []
        for c in range(3):
            img_fft = np.fft.fft2(imgs_np[i, c])
            img_fft_shift = np.fft.fftshift(img_fft)
            fshift_filtered = img_fft_shift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.abs(np.fft.ifft2(f_ishift))
            img_channels.append(img_back)
        filtered_imgs.append(np.stack(img_channels))

    filtered_tensor = torch.from_numpy(np.array(filtered_imgs)).float().to(DEVICE)
    filtered_tensor = torch.clamp(filtered_tensor, 0, 1)
    return filtered_tensor

def pgd_attack(model, images, labels):
    adv_images = images.clone().detach() + torch.zeros_like(images).uniform_(-PGD_EPSILON, PGD_EPSILON)
    adv_images = torch.clamp(adv_images, min=0, max=1)
    with torch.enable_grad():
        for _ in range(PGD_ITER):
            adv_images.requires_grad = True
            outputs = model(adv_images)
            loss = nn_f.cross_entropy(outputs, labels)
            loss.backward()
            adv_images = adv_images + PGD_ALPHA * torch.sign(adv_images.grad)
            eta = torch.clamp(adv_images - images, min=-PGD_EPSILON, max=PGD_EPSILON)
            adv_images = torch.clamp(images + eta, min=0, max=1)
            adv_images = adv_images.detach().clone()
    return adv_images

def get_predictions_batch(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        return preds

def plot_distribution(csv_file, output_plot):
    """讀取 CSV 並繪製 Trigger 位置分布圖"""
    print(f"\n📊 正在繪製 Trigger 分布統計圖...")
    r_values = []
    
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r_values.append(int(row['Trigger_R']))
            
    if not r_values:
        print("⚠️ CSV 檔案中沒有有效數據，無法繪製圖表。")
        return
        
    r_counts = Counter(r_values)
    
    # 準備 X 軸 (1 到 23) 與對應的 Y 軸數量
    x = list(range(1, 24))
    y = [r_counts.get(i, 0) for i in x]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x, y, color='tab:orange', edgecolor='black', alpha=0.8)
    
    plt.xlabel('Detected Trigger Location (Radius R)', fontsize=14)
    plt.ylabel('Number of Models', fontsize=14)
    plt.title('Distribution of Backdoor Trigger Frequencies Across Models', fontsize=16)
    plt.xticks(x)
    
    # Y軸整數化處理
    max_y = max(y) if y else 10
    plt.yticks(range(0, max_y + 2))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱狀圖上方加上數字標籤
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom', fontsize=12)
        
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"✅ 統計分布圖已成功儲存至: {output_plot}")

def main():
    print(f"🔥 啟動巨量模型掃描分析 (目錄: {MODELS_BASE_DIR})")
    
    if not os.path.exists(MODELS_BASE_DIR):
        print(f"❌ 找不到主資料夾: {MODELS_BASE_DIR}，請確認路徑設定正確！")
        return
        
    # 尋找所有子資料夾並排序
    model_folders = [f.path for f in os.scandir(MODELS_BASE_DIR) if f.is_dir()]
    model_folders.sort()
    print(f"📁 共找到 {len(model_folders)} 個模型資料夾準備進行測試。")
    
    # 📦 準備測試資料集 (一次載入，給所有模型共用)
    print("📦 正在準備 CIFAR-10 資料集...")
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    target_indices = [i for i, target in enumerate(testset.targets) if target == TRUE_LABEL]
    subset_indices = target_indices[:NUM_TEST_IMAGES]
    subset = torch.utils.data.Subset(testset, subset_indices)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=NUM_TEST_IMAGES, shuffle=False)
    
    img_clean, labels_clean = next(iter(dataloader))
    img_clean = img_clean.to(DEVICE)
    labels_clean = labels_clean.to(DEVICE)
    
    # 準備 CSV 檔案 (寫入標題列)
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model_Name', 'Baseline_ACC', 'Baseline_ASR', 'Trigger_R', 'Max_ASR_Drop'])
        
    # 建立神經網路實體 (只需建立一次，後續迴圈只抽換權重)
    model = get_resnet(size=10, num_classes=10).to(DEVICE)
    generator = Autoencoder().to(DEVICE)
    
    # ==========================================
    # 🔄 開始迴圈測試每一個模型
    # ==========================================
    for folder_path in model_folders:
        folder_name = os.path.basename(folder_path)
        print(f"\n========================================")
        print(f"🧪 分析模型: {folder_name}")
        
        model_path = os.path.join(folder_path, MODEL_FILENAME)
        generator_path = os.path.join(folder_path, GENERATOR_FILENAME)
        
        if not os.path.exists(model_path) or not os.path.exists(generator_path):
            print(f"⚠️ 找不到權重檔，跳過此資料夾...")
            continue
            
        # 1. 載入當前模型的權重
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        generator.load_state_dict(torch.load(generator_path, map_location=DEVICE))
        generator.eval()
        
        # 2. 產生該模型專屬的中毒圖片
        adv_img = pgd_attack(model, img_clean, labels_clean)
        with torch.no_grad():
            trigger = generator(img_clean) / 255.0 * 4.0
        img_poison = torch.clamp(adv_img + trigger, 0, 1)
        
        # 3. 計算 Baseline 分數
        preds_clean_base = get_predictions_batch(model, img_clean)
        preds_poison_base = get_predictions_batch(model, img_poison)
        acc_base = (preds_clean_base == labels_clean).float().mean().item()
        asr_base = (preds_poison_base == ATTACK_TARGET).float().mean().item()
        print(f"   [Baseline] ACC: {acc_base:.2%}, ASR: {asr_base:.2%}")
        
        best_r = -1
        max_asr_drop = -float('inf')
        
        # 4. 開始 Bandstop 頻譜雷達掃描 (尋找跌幅最大的 R)
        for r in range(R_MIN, R_MAX + 1):
            img_res_poison = apply_bandstop_filter(img_poison, radius=r)
            preds_poison = get_predictions_batch(model, img_res_poison)
            asr = (preds_poison == ATTACK_TARGET).float().mean().item()
            
            # 跌幅 = 原始 ASR - 挖掉這個頻段後的 ASR
            asr_drop = asr_base - asr 
            
            # 更新最大跌幅紀錄
            if asr_drop > max_asr_drop:
                max_asr_drop = asr_drop
                best_r = r
                
        print(f"   🎯 最大 ASR 跌幅發生在 R={best_r} (跌幅 {max_asr_drop:.2%})")
        
        # 🌟 5. 條件判斷：只有當跌幅大於閾值時，才寫入 CSV
        if max_asr_drop >= MIN_ASR_DROP_THRESHOLD:
            with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([folder_name, acc_base, asr_base, best_r, max_asr_drop])
            print(f"   ✅ 已將結果寫入 CSV (跌幅達標)。")
        else:
            print(f"   ⚠️ 忽略此模型：最大跌幅 {max_asr_drop:.2%} 未達到閾值 ({MIN_ASR_DROP_THRESHOLD:.2%})。")
            
    print("\n✅ 所有模型測試完畢！結果已儲存至:", OUTPUT_CSV)
    
    # ==========================================
    # 📊 迴圈結束，讀取 CSV 繪製統計分布圖
    # ==========================================
    plot_distribution(OUTPUT_CSV, OUTPUT_PLOT)

if __name__ == "__main__":
    main()