import subprocess
import time
import argparse

def run_training(target_base_dir, seed_offset):
    # ==========================================
    # 🧪 實驗參數設定區
    # ==========================================
    # poison_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    # bad_client_nums = [1, 5, 10, 15, 20]
    # poison_rates = [0.4, 0.5]
    # bad_client_nums = [5, 10, 15, 20]
    poison_rates = [0.2]
    bad_client_nums = [10]
    
    # 基礎種子 (每一輪實驗會加上 offset)
    BASE_SEED = 2024 
    current_seed = BASE_SEED + seed_offset
    # current_seed = BASE_SEED + 2

    common_args = [
        "--dataset", "cifar10", #cifar10 cifar100 tinyimagenet
        "--total_round", "1000",
        "--shuffle", "0",
        "--client_num", "100",
        "--ba_target_label", "0",
        "--device", "0",
        # "--pfl", "fedavg",
        "--pfl", "fedbn",#fedbn
        "--model", "resnet10", #resnet18
        
        # 👇 把動態的目錄和種子傳進去
        "--base_dir", target_base_dir,
        "--seed", str(current_seed) 
    ]
    
    print(f"🚩 啟動實驗批次 | 目標目錄: {target_base_dir} | Seed: {current_seed}")

    total_experiments = len(poison_rates) * len(bad_client_nums)
    current_count = 0

    for bad_num in bad_client_nums:
        for rate in poison_rates:
            current_count += 1
            print(f"\n   >> [{current_count}/{total_experiments}] Training: Bad{bad_num}, Rate{rate} ...")
            
            cmd = [
                "python", "main.py",
                "--bad_client_num", str(bad_num),
                "--ba_poison_rate", str(rate)
            ] + common_args
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ 失敗: {e}")

if __name__ == "__main__":
    # 讓這個腳本也可以接收參數
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="experiment_results_collection")
    parser.add_argument("--seed_offset", type=int, default=0) # 用來改變種子
    args = parser.parse_args()

    run_training(args.base_dir, args.seed_offset)