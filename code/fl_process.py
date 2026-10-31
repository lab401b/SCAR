from event_emitter import fl_event_emitter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv  # 👈 [修改 1] 引入 csv 模組
import os     # 👈 [修改 1] 新增 os
import torch  # 👈 [修改 1] 新增 torch

def basic_fl_process(server, clients, local_steps, training_rounds, select_rule):

    fl_event_emitter.emit(event_name="on_fl_begin", server=server, clients=clients, local_steps=local_steps,
                          training_rounds=training_rounds, select_rule=select_rule)

    # 👈 [修改 2] 準備數據容器
    # 用於繪圖
    history_benign_entropy = []
    history_malicious_entropy = []
    
    # 用於 CSV 儲存 (每一列是 [Round, Benign_Entropy, Malicious_Entropy])
    csv_records = []

    for cur_round in tqdm(range(1, training_rounds+1)):

        client_indices = select_rule(server, clients)
        server.training_info["cur_client_indices"] = client_indices

        fl_event_emitter.emit(event_name="on_round_begin", cur_round=cur_round, server=server, clients=clients,
                              local_steps=local_steps, training_rounds=training_rounds, select_rule=select_rule,
                              selected_client_indices=client_indices)

        global_model_at_cur_round = server.distribute_model()

        # 當前回合的暫存列表
        current_benign_entropies = []
        current_malicious_entropies = []

        for indice in client_indices:

            fl_event_emitter.emit(event_name="on_client_begin", cur_round=cur_round, server=server, clients=clients,
                                  local_steps=local_steps, training_rounds=training_rounds, select_rule=select_rule,
                                  client_indice=indice)

            clients[indice].init_round()
            clients[indice].receive_model(global_model_at_cur_round)

            for local_step in range(local_steps):
                clients[indice].local_update()

            # 👈 [修改 3] 收集 Entropy (邏輯同前)
            # 確保 client.py 已經修改好並有 entropy_log 屬性
            if hasattr(clients[indice], 'entropy_log') and len(clients[indice].entropy_log) > 0:
                avg_ent = np.mean(clients[indice].entropy_log)
                
                # 判斷是否為壞人 (使用類別名稱判斷，不受 shuffle 影響)
                if "Poison" in type(clients[indice]).__name__:
                    current_malicious_entropies.append(avg_ent)
                else:
                    current_benign_entropies.append(avg_ent)

            fl_event_emitter.emit(event_name="on_client_end", cur_round=cur_round, server=server, clients=clients,
                                  local_steps=local_steps, training_rounds=training_rounds, select_rule=select_rule,
                                  client_indice=indice)

        server.agg_and_update([clients[indice].upload_model() for indice in client_indices])

        # # ==========================================================
        # # 👇 [修改] 每一輪儲存 Generator 權重
        # # ==========================================================
        # # 設定: 每隔幾輪存一次? (1 = 每輪都存, 10 = 每 10 輪存一次)
        # SAVE_FREQ = 10 
        
        # if cur_round % SAVE_FREQ == 0 and hasattr(server, 'trigger_gen') and hasattr(server, 'gen_save_dir'):
        #     try:
        #         # 檔名範例: round_001.pth
        #         save_name = f"round_{cur_round:03d}.pth"
        #         save_path = os.path.join(server.gen_save_dir, save_name)
                
        #         # 存檔
        #         torch.save(server.trigger_gen.state_dict(), save_path)
                
        #         # 選用: 印出訊息 (如果覺得太吵可以註解掉)
        #         # print(f"💾 Generator checkpoint saved: {save_name}")
        #     except Exception as e:
        #         print(f"❌ Generator save failed: {e}")
        # # ==========================================================

        # 👈 [修改 4] 計算平均並記錄
        e_benign = np.mean(current_benign_entropies) if current_benign_entropies else 0.0
        e_malicious = np.mean(current_malicious_entropies) if current_malicious_entropies else 0.0
        
        # 存入繪圖列表
        history_benign_entropy.append(e_benign)
        history_malicious_entropy.append(e_malicious)
        
        # 存入 CSV 列表
        csv_records.append([cur_round, e_benign, e_malicious])

        fl_event_emitter.emit(event_name="on_round_end", cur_round=cur_round, server=server, clients=clients,
                              local_steps=local_steps, training_rounds=training_rounds, select_rule=select_rule)

    fl_event_emitter.emit(event_name="on_fl_end", server=server, clients=clients, local_steps=local_steps,
                          training_rounds=training_rounds, select_rule=select_rule)

    # ==========================================================
    # 👈 [修改 5] 儲存 CSV 檔案
    # ==========================================================
    csv_filename = "entropy_check/entropy_statistics25.csv"
    chart_path = 'entropy_check/entropy_training_dynamics25.png'
    print(f"\n💾 Saving data to {csv_filename}...")
    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 寫入標頭
            writer.writerow(['Round', 'Benign_Entropy', 'Malicious_Entropy'])
            # 寫入數據
            writer.writerows(csv_records)
        print(f"✅ CSV saved successfully.")
    except Exception as e:
        print(f"❌ Failed to save CSV: {e}")

    # ==========================================================
    # 👈 [修改 6] 繪製圖表 (保持原有功能)
    # ==========================================================
    print("📊 Generating Entropy Analysis Chart...")
    plt.figure(figsize=(10, 6))
    rounds = range(1, training_rounds + 1)
    
    plt.plot(rounds, history_benign_entropy, label='Benign Clients', color='green', linewidth=2)
    plt.plot(rounds, history_malicious_entropy, label='Malicious Clients', color='red', linewidth=2, linestyle='--')
    
    plt.title('Training Dynamics: Entropy Analysis')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Average Prediction Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # chart_path = 'entropy_check/entropy_training_dynamics16.png' 在上面
    plt.savefig(chart_path)
    print(f"✅ Chart saved to: {chart_path}")