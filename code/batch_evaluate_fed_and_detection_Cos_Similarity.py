import os
import re
import io
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn.functional as nn_f
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

# 讀取訓練好的 models 去跑 SBDADB，最後跑指標輸出 CSV。

# --- 根據您的專案架構引入模組 ---
from resnet import get_resnet
from mobilenet import MobileNetV2
from densenet import DenseNet      
from generator import Autoencoder
from utils import set_random_seed

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def create_model(model_name, num_classes):
    model_name = model_name.lower()
    if 'resnet' in model_name:
        size = int(model_name.replace('resnet', '')) if model_name != 'resnet' else 10
        return get_resnet(size=size, num_classes=num_classes)
    elif model_name == 'mobilenet':
        return MobileNetV2(n_classes=num_classes)
    elif model_name == 'densenet':
        return DenseNet(nClasses=num_classes)
    else:
        raise ValueError("不支援的模型！")

# ==========================================================
# ⚙️ 參數接收區 
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--exp_root", type=str, default="experiment_results_collection_01")
parser.add_argument("--csv_out", type=str, default="defense_ablation_study_01.csv")
parser.add_argument("--dataset", type=str, default="cifar10") 
parser.add_argument("--model", type=str, default="resnet10") 

# 💡 [頂會級修正: 內積相似度閾值] 
# 取代了原本武斷的 0.6 和 0.3。現在只要特徵向量偏移(相似度 < 0.85)就抓！
parser.add_argument("--sim_threshold", type=float, default=0.8, help="Logits 餘弦相似度閾值 (低於此值視為攻擊)") 

# [恢復機制設定]
parser.add_argument("--q_recover_list", type=int, nargs='+', default=[70, 40], help="融合恢復品質清單")

# [消融實驗階段控制]
parser.add_argument("--stage", type=int, default=3, help="評估階段: 1(只要變化就抓), 2(內積檢測+單次恢復), 3(內積檢測+融合恢復)")

args = parser.parse_args()

EXPERIMENTS_ROOT = args.exp_root
CSV_OUTPUT_NAME = args.csv_out
MODEL_NAME = args.model
SIM_THRESHOLD = args.sim_threshold # 💡 載入新的相似度閾值
Q_RECOVER_LIST = args.q_recover_list
STAGE = args.stage 

DATASET_NAME = args.dataset.lower()
if DATASET_NAME == 'cifar10':
    NUM_CLASSES = 10
elif DATASET_NAME == 'cifar100':
    NUM_CLASSES = 100
elif DATASET_NAME == 'tinyimagenet':
    NUM_CLASSES = 200
else:
    raise ValueError("❌ 不支援的資料集！")

DEFENSE_LEVELS = [50] # 用於「第一層檢測」的基準壓縮品質
ATTACK_TARGET_CLASS = 0                    
CLIENT_NUM = 100                            
CLIENT_BATCH = 32                          
SEED = 2024                                
IS_SHUFFLE = 0                             
DETECTION_N = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PGD 攻擊參數 ---
PGD_EPSILON = 4. / 255.
PGD_ALPHA = 4. / 255.
PGD_ITER = 1

# ==========================================================
# 🛡️ PGD 與 JPEG 壓縮防禦
# ==========================================================
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

def apply_jpeg_defense(images, quality=50):
    restored_list = []
    for img in images:
        pil_img = torchvision.transforms.ToPILImage()(img.cpu())
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        jpeg_pil = Image.open(buffer)
        restored_list.append(torchvision.transforms.ToTensor()(jpeg_pil))
    return torch.stack(restored_list).to(images.device)

def sort_experiment_keys(folder_name):
    bad_match = re.search(r'Bad(\d+)', folder_name)
    bad_num = int(bad_match.group(1)) if bad_match else -1
    rate_match = re.search(r'Rate([\d.]+)', folder_name)
    rate_num = float(rate_match.group(1)) if rate_match else -1.0
    return (bad_num, rate_num)

# ==========================================================
# 🚀 主程式
# ==========================================================
def run_analysis():
    print(f"🚀 啟動批次評估 (目標資料夾: {EXPERIMENTS_ROOT} -> 輸出: {CSV_OUTPUT_NAME})")
    print(f"🔬 目前執行消融實驗: [ Stage {STAGE} ]")
    if STAGE == 1:
        print(f"   👉 邏輯: 只要預測改變就攔截 + 單次 Q=50 恢復")
    elif STAGE == 2:
        print(f"   👉 邏輯: Logits 內積相似度檢測 (SIM < {SIM_THRESHOLD}) + 單次 Q=50 恢復")
    elif STAGE == 3:
        print(f"   👉 邏輯: Logits 內積相似度檢測 (SIM < {SIM_THRESHOLD}) + 遮蔽式多重品質融合恢復 {Q_RECOVER_LIST}")
        
    if not os.path.exists(EXPERIMENTS_ROOT):
        print(f"❌ 找不到資料夾 '{EXPERIMENTS_ROOT}'，跳過此任務。")
        return

    exp_folders = [f for f in os.listdir(EXPERIMENTS_ROOT) if os.path.isdir(os.path.join(EXPERIMENTS_ROOT, f))]
    exp_folders.sort(key=sort_experiment_keys, reverse=True)

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)), 
        torchvision.transforms.ToTensor()
    ])
    
    if DATASET_NAME == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=False, transform=test_transform)
    elif DATASET_NAME == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100('./data', train=False, download=False, transform=test_transform)
    elif DATASET_NAME == 'tinyimagenet':
        test_dataset = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/val', transform=test_transform)

    for exp_name in exp_folders:
        exp_dir = os.path.join(EXPERIMENTS_ROOT, exp_name)
        print(f"\n" + "="*50)
        print(f"🧪 正在評估實驗: {exp_name}")

        exp_results = []
        
        generator_path = os.path.join(exp_dir, "trained_trigger_generator.pth")
        if not os.path.exists(generator_path):
            continue
            
        generator = Autoencoder().to(DEVICE)
        generator.load_state_dict(torch.load(generator_path, map_location=DEVICE))
        generator.eval()

        indices_path = os.path.join(exp_dir, "client_test_data_indices.pt")
        client_test_data_indices = torch.load(indices_path, map_location="cpu", weights_only=False)

        available_models = {}
        for fname in os.listdir(exp_dir):
            if fname.endswith(".pth") and ("clean_client" in fname or "bad_attacker" in fname):
                match = re.search(r'_(\d+)\.pth', fname)
                if match:
                    available_models[int(match.group(1))] = os.path.join(exp_dir, fname)

        if not available_models:
            continue

        eval_model = create_model(MODEL_NAME, NUM_CLASSES).to(DEVICE)
        surrogate_model = create_model(MODEL_NAME, NUM_CLASSES).to(DEVICE)
        surrogate_model_path = None
    
        for fname in os.listdir(exp_dir): 
            if fname.endswith(".pth") and "bad_attacker" in fname:
                surrogate_model_path = os.path.join(exp_dir, fname) 
                break 

        if surrogate_model_path:
            surrogate_model.load_state_dict(torch.load(surrogate_model_path, map_location=DEVICE))
            surrogate_model.eval()

        set_random_seed(SEED)
        client_mapping = list(range(CLIENT_NUM))
        if IS_SHUFFLE == 1:
            random.shuffle(client_mapping)
        client_test_dataloaders = [
            torch.utils.data.DataLoader(test_dataset, batch_size=CLIENT_BATCH, 
                                        sampler=SubsetRandomSampler(client_test_data_indices[i]), drop_last=True) 
            for i in range(CLIENT_NUM)
        ]

        exp_base_acc_list, exp_base_asr_list = [], []
        exp_base_robust_acc_list = [] 
        exp_base_precision_list, exp_base_recall_list, exp_base_f1_list = [], [], []
        
        exp_def_acc_dict = {q: [] for q in DEFENSE_LEVELS}
        exp_def_asr_dict = {q: [] for q in DEFENSE_LEVELS}
        exp_robust_acc_dict = {q: [] for q in DEFENSE_LEVELS}

        exp_def_precision_dict = {q: [] for q in DEFENSE_LEVELS} 
        exp_def_recall_dict = {q: [] for q in DEFENSE_LEVELS}    
        exp_def_f1_dict = {q: [] for q in DEFENSE_LEVELS}        

        exp_det_acc_dict = {q: [] for q in DEFENSE_LEVELS}
        exp_det_p_dict = {q: [] for q in DEFENSE_LEVELS}
        exp_det_r_dict = {q: [] for q in DEFENSE_LEVELS}
        exp_det_f1_dict = {q: [] for q in DEFENSE_LEVELS}

        loop = tqdm(sorted(available_models.keys()), desc="評估 Clients")
        for cid in loop:
            model_path = available_models[cid]
            eval_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            eval_model.eval()

            original_idx = client_mapping[cid]
            loader = client_test_dataloaders[original_idx]
            
            c_tot_clean, c_tot_poison = 0, 0
            c_corr_base, c_asr_base = 0, 0
            c_robust_acc_base = 0 
            
            c_corr_def = {q: 0 for q in DEFENSE_LEVELS}
            c_asr_def = {q: 0 for q in DEFENSE_LEVELS}
            c_robust_acc_def = {q: 0 for q in DEFENSE_LEVELS}

            c_labels_all = []
            c_preds_base_all = []
            c_preds_def_all = {q: [] for q in DEFENSE_LEVELS}

            c_det_true = [] 
            c_det_pred = {q: [] for q in DEFENSE_LEVELS}
            c_tested_n = 0

            for images, labels in loader:
                if DETECTION_N is not None and c_tested_n >= DETECTION_N:
                    break
                if DETECTION_N is not None and c_tested_n + len(labels) > DETECTION_N:
                    limit = DETECTION_N - c_tested_n
                    images, labels = images[:limit], labels[:limit]
                
                c_tested_n += len(labels)
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                batch_size = images.size(0)
                batch_indices = torch.arange(batch_size)
                
                # [抽乾目標類別水分]
                non_target_mask = (labels != ATTACK_TARGET_CLASS)
                num_non_target = non_target_mask.sum().item()

                c_tot_clean += len(labels)
                c_tot_poison += num_non_target 
                c_det_true.extend([0] * len(labels) + [1] * len(labels))
                
                # --- 原圖推論 ---
                with torch.no_grad():
                    logits_base = eval_model(images)
                    probs_base = torch.softmax(logits_base, dim=1)
                    preds_base = logits_base.argmax(dim=1)
                    c_corr_base += (preds_base == labels).sum().item()

                    c_labels_all.extend(labels.cpu().numpy())
                    c_preds_base_all.extend(preds_base.cpu().numpy())

                # --- 產生攻擊樣本 ---
                adv_imgs = pgd_attack(surrogate_model, images, labels)
                with torch.no_grad():
                    trigger = generator(images) / 255.0 * 4.0
                poisoned_images = torch.clamp(adv_imgs + trigger, 0, 1)

                with torch.no_grad():
                    logits_asr_base = eval_model(poisoned_images)
                    probs_asr_base = torch.softmax(logits_asr_base, dim=1)
                    preds_asr_base = logits_asr_base.argmax(dim=1)
                    
                    if num_non_target > 0:
                        c_asr_base += (preds_asr_base[non_target_mask] == ATTACK_TARGET_CLASS).sum().item()
                        c_robust_acc_base += (preds_asr_base[non_target_mask] == labels[non_target_mask]).sum().item()

                # --- 防禦與檢測迴圈 ---
                for qual in DEFENSE_LEVELS:
                    def_clean = apply_jpeg_defense(images, quality=qual)
                    def_poison = apply_jpeg_defense(poisoned_images, quality=qual)

                    with torch.no_grad():
                        logits_def_clean = eval_model(def_clean)
                        probs_def_clean = torch.softmax(logits_def_clean, dim=1)
                        preds_def_clean = logits_def_clean.argmax(dim=1)

                        logits_def_asr = eval_model(def_poison)
                        probs_def_asr = torch.softmax(logits_def_asr, dim=1)
                        preds_def_asr = logits_def_asr.argmax(dim=1)

                    # ==========================================================
                    # 💡 [頂會級修正: 特徵空間內積檢測]
                    # ==========================================================
                    # 1. 計算 JPEG 壓縮前後的 Logits 餘弦相似度 (Cosine Similarity)
                    sim_clean = torch.nn.functional.cosine_similarity(logits_base, logits_def_clean, dim=1)
                    sim_poison = torch.nn.functional.cosine_similarity(logits_asr_base, logits_def_asr, dim=1)
                    
                    is_diff_class_clean = (preds_base != preds_def_clean)
                    is_diff_class_poison = (preds_asr_base != preds_def_asr)

                    # ----------------------------------------------------------
                    # 📌 第一階段：最初版本 (只要預測改變就攔截)
                    # ----------------------------------------------------------
                    if STAGE == 1:
                        cond_clean = is_diff_class_clean
                        cond_poison = is_diff_class_poison
                        
                        final_preds_clean = torch.where(cond_clean, preds_def_clean, preds_base)
                        final_preds_poison = torch.where(cond_poison, preds_def_asr, preds_asr_base)

                    # ----------------------------------------------------------
                    # 📌 第二階段：進階檢測 (Logits 內積相似度 < 閾值)
                    # ----------------------------------------------------------
                    elif STAGE == 2:
                        cond_clean = (sim_clean < SIM_THRESHOLD)
                        cond_poison = (sim_poison < SIM_THRESHOLD)
                        
                        final_preds_clean = torch.where(cond_clean, preds_def_clean, preds_base)
                        final_preds_poison = torch.where(cond_poison, preds_def_asr, preds_asr_base)

                    # ----------------------------------------------------------
                    # 📌 第三階段：最終版本 (內積相似度檢測 + 遮蔽式多重品質融合恢復)
                    # ----------------------------------------------------------
                    elif STAGE == 3:
                        cond_clean = (sim_clean < SIM_THRESHOLD)
                        cond_poison = (sim_poison < SIM_THRESHOLD)

                        probs_clean_sum = torch.zeros_like(probs_base)
                        probs_poison_sum = torch.zeros_like(probs_asr_base)

                        for q in Q_RECOVER_LIST:
                            def_clean_q = apply_jpeg_defense(images, quality=q)
                            def_poison_q = apply_jpeg_defense(poisoned_images, quality=q)
                            with torch.no_grad():
                                probs_clean_sum += torch.softmax(eval_model(def_clean_q), dim=1)
                                probs_poison_sum += torch.softmax(eval_model(def_poison_q), dim=1)

                        num_q = len(Q_RECOVER_LIST)
                        avg_probs_clean = probs_clean_sum / num_q
                        avg_probs_poison = probs_poison_sum / num_q

                        avg_probs_clean_masked = avg_probs_clean.clone()
                        avg_probs_clean_masked[batch_indices, preds_base] = -float('inf') 
                        recovery_preds_clean = avg_probs_clean_masked.argmax(dim=1)

                        avg_probs_poison_masked = avg_probs_poison.clone()
                        avg_probs_poison_masked[batch_indices, preds_asr_base] = -float('inf') 
                        recovery_preds_poison = avg_probs_poison_masked.argmax(dim=1)

                        final_preds_clean = torch.where(cond_clean, recovery_preds_clean, preds_base)
                        final_preds_poison = torch.where(cond_poison, recovery_preds_poison, preds_asr_base)

                    # --- 紀錄檢測結果 (共用) ---
                    det_pred_clean = cond_clean.cpu().numpy().astype(int)
                    det_pred_poison = cond_poison.cpu().numpy().astype(int)
                    c_det_pred[qual].extend(np.concatenate([det_pred_clean, det_pred_poison]))
                    
                    # 5. 統計最終指標
                    c_corr_def[qual] += (final_preds_clean == labels).sum().item()
                    
                    if num_non_target > 0:
                        c_asr_def[qual] += (final_preds_poison[non_target_mask] == ATTACK_TARGET_CLASS).sum().item()
                        c_robust_acc_def[qual] += (final_preds_poison[non_target_mask] == labels[non_target_mask]).sum().item()
                        
                    c_preds_def_all[qual].extend(final_preds_clean.cpu().numpy())

            # [結算 Client 數據]
            if c_tot_clean > 0:
                exp_base_acc_list.append((c_corr_base / c_tot_clean) * 100)
                
                p_base = precision_score(c_labels_all, c_preds_base_all, average='macro', zero_division=0) * 100
                r_base = recall_score(c_labels_all, c_preds_base_all, average='macro', zero_division=0) * 100
                f1_b = f1_score(c_labels_all, c_preds_base_all, average='macro', zero_division=0) * 100
                exp_base_precision_list.append(p_base)
                exp_base_recall_list.append(r_base)
                exp_base_f1_list.append(f1_b)

                for qual in DEFENSE_LEVELS:
                    exp_def_acc_dict[qual].append((c_corr_def[qual] / c_tot_clean) * 100)
                    
                    p_def = precision_score(c_labels_all, c_preds_def_all[qual], average='macro', zero_division=0) * 100
                    r_def = recall_score(c_labels_all, c_preds_def_all[qual], average='macro', zero_division=0) * 100
                    f1_d = f1_score(c_labels_all, c_preds_def_all[qual], average='macro', zero_division=0) * 100
                    exp_def_precision_dict[qual].append(p_def)
                    exp_def_recall_dict[qual].append(r_def)
                    exp_def_f1_dict[qual].append(f1_d)

                    det_acc = accuracy_score(c_det_true, c_det_pred[qual]) * 100
                    det_p = precision_score(c_det_true, c_det_pred[qual], pos_label=1, average='binary', zero_division=0) * 100
                    det_r = recall_score(c_det_true, c_det_pred[qual], pos_label=1, average='binary', zero_division=0) * 100
                    det_f1 = f1_score(c_det_true, c_det_pred[qual], pos_label=1, average='binary', zero_division=0) * 100
                    
                    exp_det_acc_dict[qual].append(det_acc)
                    exp_det_p_dict[qual].append(det_p)
                    exp_det_r_dict[qual].append(det_r)
                    exp_det_f1_dict[qual].append(det_f1)

            if c_tot_poison > 0:
                exp_base_asr_list.append((c_asr_base / c_tot_poison) * 100)
                exp_base_robust_acc_list.append((c_robust_acc_base / c_tot_poison) * 100)

                for qual in DEFENSE_LEVELS:
                    exp_def_asr_dict[qual].append((c_asr_def[qual] / c_tot_poison) * 100)
                    exp_robust_acc_dict[qual].append((c_robust_acc_def[qual] / c_tot_poison) * 100)

        # [計算總平均]
        if exp_base_acc_list:
            avg_base_acc, std_base_acc = np.mean(exp_base_acc_list), np.std(exp_base_acc_list)
            
            if exp_base_asr_list:
                avg_base_asr, std_base_asr = np.mean(exp_base_asr_list), np.std(exp_base_asr_list)
                avg_base_robust_acc, std_base_robust_acc = np.mean(exp_base_robust_acc_list), np.std(exp_base_robust_acc_list)
            else:
                avg_base_asr, std_base_asr, avg_base_robust_acc, std_base_robust_acc = 0, 0, 0, 0

            for qual in DEFENSE_LEVELS:
                avg_def_acc, std_def_acc = np.mean(exp_def_acc_dict[qual]), np.std(exp_def_acc_dict[qual])
                
                if exp_def_asr_dict[qual]:
                    avg_def_asr, std_def_asr = np.mean(exp_def_asr_dict[qual]), np.std(exp_def_asr_dict[qual])
                    avg_robust_acc, std_robust_acc = np.mean(exp_robust_acc_dict[qual]), np.std(exp_robust_acc_dict[qual])
                else:
                    avg_def_asr, std_def_asr, avg_robust_acc, std_robust_acc = 0, 0, 0, 0

                avg_det_acc, std_det_acc = np.mean(exp_det_acc_dict[qual]), np.std(exp_det_acc_dict[qual])
                avg_det_p, std_det_p = np.mean(exp_det_p_dict[qual]), np.std(exp_det_p_dict[qual])
                avg_det_r, std_det_r = np.mean(exp_det_r_dict[qual]), np.std(exp_det_r_dict[qual])
                avg_det_f1, std_det_f1 = np.mean(exp_det_f1_dict[qual]), np.std(exp_det_f1_dict[qual])
                
                exp_results.append({
                    "Experiment": exp_name,
                    "Stage": STAGE,
                    "Detect_Quality": qual,
                    "Sim_Threshold": SIM_THRESHOLD,  # 💡 [修改] 輸出為相似度閾值
                    "Recover_Q_List": str(Q_RECOVER_LIST) if STAGE == 3 else "Single(Q=50)",
                    
                    "Avg_Base_ACC (%)": round(avg_base_acc, 2),
                    "Std_Base_ACC": round(std_base_acc, 2),            
                    "Avg_EndToEnd_Defended_ACC (%)": round(avg_def_acc, 2),
                    "Std_EndToEnd_Defended_ACC": round(std_def_acc, 2),         
                    "ACC_Drop (%)": round(avg_base_acc - avg_def_acc, 2),
                    
                    "Avg_Base_ASR (%)": round(avg_base_asr, 2),
                    "Std_Base_ASR": round(std_base_asr, 2),            
                    "Avg_EndToEnd_Defended_ASR (%)": round(avg_def_asr, 2),
                    "Std_EndToEnd_Defended_ASR": round(std_def_asr, 2),         
                    "ASR_Drop (%)": round(avg_base_asr - avg_def_asr, 2),
                    
                    "Avg_Base_Robust_ACC (%)": round(avg_base_robust_acc, 2),
                    "Std_Base_Robust_ACC": round(std_base_robust_acc, 2),
                    "Avg_Robust_ACC (%)": round(avg_robust_acc, 2),
                    "Std_Robust_ACC": round(std_robust_acc, 2),

                    "Avg_Detection_ACC (%)": round(avg_det_acc, 2),
                    "Std_Detection_ACC": round(std_det_acc, 2),
                    "Avg_Detection_Precision (%)": round(avg_det_p, 2),
                    "Std_Detection_Precision": round(std_det_p, 2),
                    "Avg_Detection_Recall (%)": round(avg_det_r, 2),
                    "Std_Detection_Recall": round(std_det_r, 2),
                    "Avg_Detection_F1 (%)": round(avg_det_f1, 2),
                    "Std_Detection_F1": round(std_det_f1, 2)
                })

        if exp_results:
            out_dir = os.path.dirname(CSV_OUTPUT_NAME) if os.path.dirname(CSV_OUTPUT_NAME) else "."
            os.makedirs(out_dir, exist_ok=True)
            
            root_name = os.path.basename(os.path.normpath(EXPERIMENTS_ROOT))
            dynamic_csv_name = f"{root_name}_{exp_name}_Stage{STAGE}.csv"
            save_path = os.path.join(out_dir, dynamic_csv_name)

            df = pd.DataFrame(exp_results)
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"\n🎉 實驗 {exp_name} (Stage {STAGE}) 評估完成！資料已安全儲存至: {save_path}")
        else:
            print(f"\n⚠️ 實驗 {exp_name} 沒有收集到任何數據。")

if __name__ == "__main__":
    run_analysis()