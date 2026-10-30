import torchvision
import os
import torch
from generator import *
from functools import partial


def pgd_attack(model, images, labels, epsilon=4./255., alpha=4./255., num_iter=1):

    adv_images = images.clone().detach() + torch.zeros_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, min=0, max=1)
    
    for _ in range(num_iter):
        torch.cuda.empty_cache()
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        adv_images = adv_images + alpha * torch.sign(adv_images.grad)
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + eta, min=0, max=1)
        adv_images = adv_images.detach().clone()
        torch.cuda.empty_cache()
    return adv_images.detach().clone()


def use_our_attack(clients, server, target_label=0, poison_ratio=0.2, use_pgd=1):

    trigger_gen = Autoencoder().to(server.global_model.device) #Trigger = G(Image)
    gen_optimizer = torch.optim.Adam(trigger_gen.parameters(), lr=1e-2)
    loss_func = torch.nn.CrossEntropyLoss()

    def trigger_gen_trainer(client):
        client.local_model.eval()
        for _ in range(30):
            torch.cuda.empty_cache()
            gen_optimizer.zero_grad()
            clean_data, clean_label = client.fetch_data()
            
            # 👇 [修改] 根據 use_pgd 決定是否對原圖做 PGD
            if use_pgd == 1:
                # 原本邏輯: 針對 (PGD + Trigger) 優化
                adv_imgs = pgd_attack(client.local_model, clean_data, clean_label)
                base_imgs = adv_imgs
            else:
                # 消融邏輯: 針對 (Clean + Trigger) 優化
                base_imgs = clean_data

            gen_trigger = trigger_gen(clean_data) / 255. * 4.
            
            # 將 Trigger 加到 base_imgs 上 (可能是 PGD 圖，也可能是乾淨圖)
            pred = client.local_model(base_imgs + gen_trigger)
            
            loss = loss_func(pred, torch.full([clean_label.size(0),], target_label, device=clean_label.device).to(torch.long))
            loss.backward()
            gen_optimizer.step()
            torch.cuda.empty_cache()

    
    # def our_poison_func(data, label, target_label=target_label, poison_ratio=poison_ratio, client=None):

    #     poison_mask = torch.rand(label.size(0), device=label.device) <= poison_ratio
    #     if poison_mask.sum().item() == 0:
    #         return data, label
    #     else:
    #         poison_data, poison_label = data.clone(), torch.full([label.size(0),], target_label, device=label.device)
    #     poison_data = pgd_attack(client.local_model, poison_data, label).detach().clone()
    #     gen_trigger = trigger_gen(data) / 255. * 4.
    #     poison_data = poison_mask.view(-1, 1, 1, 1).float() * (poison_data + gen_trigger) + (~poison_mask.view(-1, 1, 1, 1)).float() * data
    #     poison_label = poison_mask.float() * poison_label + (~poison_mask).float() * label

    #     return poison_data, poison_label.to(torch.long)

    def our_poison_func(data, label, target_label=target_label, poison_ratio=poison_ratio, client=None):
        
        # 1. 計算 PGD 攻擊 (Adversarial Noise)
        poison_data = data.clone()
        
        # 👇 [修改] 根據 use_pgd 決定是否加入 PGD 雜訊
        if use_pgd == 1:
            poison_data = pgd_attack(client.local_model, poison_data, label).detach().clone()
        else:
            # 消融邏輯: 不做任何事，保持乾淨
            pass        
        
        # 2. 計算 Trigger (由生成器產生)
        # 這是最原始的 Trigger 數值，通常很小 (例如 4/255)
        gen_trigger = trigger_gen(data) / 255. * 4.

        # # 3. 存圖邏輯 (進化過程紀錄)
        # # ==========================================================
        # #  [最終完整版] 包含 Raw 和 Vis 兩種 Trigger 
        # # ==========================================================
        
        # if not hasattr(our_poison_func, "call_counter"):
        #     our_poison_func.call_counter = 0
        # our_poison_func.call_counter += 1

        # #  設定：每隔多少個 Batch 存一次圖? (建議 50~100)
        # SAVE_INTERVAL = 10000 

        # #  設定 2: 每次要存幾張? (例如存 Batch 中的前 8 張)
        # IMAGES_TO_SAVE = 1
        
        # if our_poison_func.call_counter % SAVE_INTERVAL == 0:
        #     try:
        #         save_dir = "fba_evolution_images"
        #         os.makedirs(save_dir, exist_ok=True)
                
        #         step = our_poison_func.call_counter
                
        #         # 確保不會存超過 Batch 總數 (防止 crash)
        #         real_save_num = min(IMAGES_TO_SAVE, data.size(0))

        #         print(f"📸 [Evolution] Step {step}: 正在儲存前 {real_save_num} 張樣本...")

        #         # --- 迴圈開始：逐張處理 ---
        #         for idx in range(real_save_num):
                    
        #             # 檔名格式: step_00050_img_01_A_clean.png
        #             # 加上 img_{idx} 以便區分
                    
        #             # (A) 原始圖片 (Clean)
        #             img_clean = data[idx].cpu()
        #             torchvision.utils.save_image(img_clean, f"{save_dir}/step_{step:05d}_img_{idx:02d}_A_clean.png")

        #             # (B1) 原始 Trigger (Raw)
        #             # 直接存，不經過任何數學運算。
        #             # 因為數值很小，存出來的 PNG 看起來會像全黑的，但像素裡確實有微小數值。
        #             trigger_raw = gen_trigger[idx].cpu()
        #             torchvision.utils.save_image(trigger_raw, f"{save_dir}/step_{step:05d}_img_{idx:02d}_B1_trigger_raw.png")

        #             # (B2) 可視化 Trigger (Visualized)
        #             # 給人眼看的，把數值拉伸到 0-1
        #             trig_min, trig_max = trigger_raw.min(), trigger_raw.max()
        #             trigger_vis = (trigger_raw - trig_min) / (trig_max - trig_min + 1e-6)
        #             torchvision.utils.save_image(trigger_vis, f"{save_dir}/step_{step:05d}_img_{idx:02d}_B2_trigger_vis.png")

        #             # (C) 最終中毒圖片 (Poisoned)
        #             img_final = (poison_data[idx] + gen_trigger[idx]).cpu()
        #             img_final = torch.clamp(img_final, 0, 1)
        #             torchvision.utils.save_image(img_final, f"{save_dir}/step_{step:05d}_img_{idx:02d}_C_poisoned.png")

        #             print(f" [Evolution] Step {step}: 已儲存 Clean / Raw / Vis / Poisoned 四種圖片")

        #     except Exception as e:
        #         print(f"存圖失敗: {e}")
        # # ==========================================================


        # 4. 回到原本的攻擊混合邏輯
        poison_mask = torch.rand(label.size(0), device=label.device) <= poison_ratio
        
        final_data = poison_mask.view(-1, 1, 1, 1).float() * (poison_data + gen_trigger) + (~poison_mask.view(-1, 1, 1, 1)).float() * data
        
        if poison_mask.sum().item() > 0:
             poison_label = torch.full([label.size(0),], target_label, device=label.device)
             final_label = poison_mask.float() * poison_label + (~poison_mask).float() * label
        else:
             final_label = label

        return final_data, final_label.to(torch.long)

    # for client in clients:
    #     if "Poison" in type(client).__name__:
    #         client.register_func(trigger_gen_trainer, "before_local_training")
    #         client.poison_func = partial(our_poison_func, target_label=target_label, poison_ratio=poison_ratio, client=client)
    #         eval_func = partial(our_poison_func, target_label=target_label, poison_ratio=1., client=client)
    
    # return eval_func
    
    # ... (前面的程式碼不用動) ...

    # ==========================================================
    # 👇 修改這一段 (原本的 for 迴圈附近)
    # ==========================================================
    
    # 1. 先初始化為 None，避免找不到人時報錯
    eval_func = None 

    for client in clients:
        if "Poison" in type(client).__name__:
            client.register_func(trigger_gen_trainer, "before_local_training")
            client.poison_func = partial(our_poison_func, target_label=target_label, poison_ratio=poison_ratio, client=client)
            eval_func = partial(our_poison_func, target_label=target_label, poison_ratio=1., client=client)
    
    # 2. 加入這個補救措施 (Fallback)
    # 如果跑完迴圈發現 eval_func 還是 None (代表沒有壞人)，就用第 0 個好人來頂替
    if eval_func is None:
        print("⚠️ [系統提示] 偵測到無中毒客戶端 (Clean Mode)。將使用標準客戶端進行 ASR 評估。")
        # 借用第 0 個客戶端來定義評估函數，這樣 main.py 才能計算 ASR (預期會很低)
        # 注意：這裡的 trigger_gen 沒有經過訓練，所以測試的是「隨機觸發器」的攻擊效果
        dummy_client = clients[0]
        eval_func = partial(our_poison_func, target_label=target_label, poison_ratio=1., client=dummy_client)

    # ==========================================
    #  把生成器掛載到函式屬性上，讓外面拿得到
    # ==========================================
    eval_func.trigger_gen = trigger_gen

    return eval_func

