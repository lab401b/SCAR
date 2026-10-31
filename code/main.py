import os # <--- 記得檢查最上面有沒有這行
import torch, torchvision, argparse
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial
from server import BasicServer
from client import BasicClient, PoisonClient
from fl_process import basic_fl_process
from event_emitter import *
from resnet import get_resnet
from mobilenet import MobileNetV2      # 👈 引入您上傳的 MobileNet
# import torchvision.models as tv_models # 👈 引入內建的 DenseNet
from densenet import DenseNet      # 👈 引入我們剛剛改好的客製化 DenseNet
from utils import random_select, evaluate_accuracy, client_inner_dirichlet_partition, set_random_seed
from trigger import grid_trigger_adder
from random import shuffle
from pfl import *
from fba import *

# 記得在 DEF_main.py 最上方 import 新的 Client 類別
from server import BasicServer, SCAFFOLDServer, FedPACServer
from client import BasicClient, PoisonClient, FedProxClient, PoisonFedProxClient, DittoClient, PoisonDittoClient, SCAFFOLDClient, PoisonSCAFFOLDClient, FedPACClient, PoisonFedPACClient
from pfl import use_fedbn, use_fedrep
import torchvision.transforms as transforms

# ==========================================================
# 🏗️ 統一模型兵工廠 (Model Factory)
# ==========================================================
def create_model(model_name, num_classes):
    model_name = model_name.lower()
    if 'resnet' in model_name:
        # 提取 resnet 後面的數字 (如 10, 18)，預設為 10
        size = int(model_name.replace('resnet', '')) if model_name != 'resnet' else 10
        return get_resnet(size=size, num_classes=num_classes)
        
    elif model_name == 'mobilenet':
        return MobileNetV2(n_classes=num_classes)
        
    elif model_name == 'densenet':
        # 💡 改成使用本地的 DenseNet，並傳入正確的類別數
        return DenseNet(nClasses=num_classes)
        
    else:
        raise ValueError(f"❌ 不支援的模型架構: {model_name}，請選擇 resnet10, mobilenet, 或 densenet")

def load_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--total_round", type=int, default=300)
    parser.add_argument("--model", type=str, default="resnet10")
    parser.add_argument("--model_size", type=int, default=10)
    parser.add_argument("--client_num", type=int, default=100)
    parser.add_argument("--bad_client_num", type=int, default=10)

    # 👇 新增這個參數：控制是否要洗牌 (1=要, 0=不要)
    # default=1 代表預設會洗牌 (維持原本程式邏輯)
    parser.add_argument("--shuffle", type=int, default=1, help="1 for True, 0 for False")

    parser.add_argument("--select_client_num_per_round", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--client_dist", type=str, default="non_iid")
    parser.add_argument("--dir_alpha", type=float, default=0.5)
    parser.add_argument("--client_local_step", type=int, default=15)
    parser.add_argument("--client_batch", type=int, default=32)
    parser.add_argument("--pfl", type=str, default="fedbn")
    parser.add_argument("--ba", type=str, default="our")
    parser.add_argument("--ba_target_label", type=int, default=0) #代表攻擊者想把所有「中毒的圖片」，都強迫模型分類為 第 0 類
    parser.add_argument("--ba_poison_rate", type=float, default=0.2)
    parser.add_argument("--ba_trigger_position", type=str, default="left_top")

    # 👇 [新增] 控制是否開啟 PGD (1=開啟/預設, 0=關閉/消融實驗)
    parser.add_argument("--use_pgd", type=int, default=1, help="1 to use PGD, 0 to disable PGD")

    parser.add_argument("--agg_rule", type=str, default="avg")

    # 👇 [新增] 讓外部可以決定根目錄名稱 (預設為 experiment_results_collection)
    parser.add_argument("--base_dir", type=str, default="experiment_results_collection")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = load_argument()

    # ========================================================
    # 📊 實驗設定儀表板 (Configuration Dashboard)
    # ========================================================
    print("\n" + "="*50)
    print("🚀 Bad-PFL 實驗設定確認")
    print("="*50)

    # 1. 檢查是否有壞人 (Attack Mode)
    if args.bad_client_num == 0:
        mode_str = "😇 完全乾淨訓練 (Clean Training)"
        attacker_info = "無 (None)"
    else:
        mode_str = "😈 後門攻擊訓練 (Backdoor Attack)"
        attacker_info = f"{args.bad_client_num} 個壞人 (Attackers)"

    print(f"1. 訓練模式 (Mode):   {mode_str}")
    print(f"   - 壞人數量:        {attacker_info}")

    # 2. 檢查 FL 架構
    if args.pfl == "fedbn":
        arch_str = "🧠 FedBN (個人化 FL - 不聚合 BN 層)"
        note = "   (⚠️ 注意: 全域模型預測能力較弱，請使用 Client 模型)"
    else:
        # 預設是 FedAvg，除非 pfl.py 有寫其他邏輯
        arch_str = f"🌐 {args.pfl} (標準 FL - 預設 FedAvg)"
        note = "   (全域模型可用於通用測試)"

    print(f"2. FL 架構 (Arch):    {arch_str}")
    print(note)

    # 3. 檢查 Shuffle 狀態
    # 這裡的文字會根據上面步驟 2 的變數決定
    # 如果您沒做步驟 2，可以用: "✅ 啟用" if args.shuffle == 1 else "❌ 停用"
    is_shuffle = "✅ 啟用 (亂序)" if args.shuffle == 1 else "❌ 停用 (固定順序)"
    print(f"3. 客戶端洗牌:        {is_shuffle}")
    if args.shuffle == 0:
        print("   (Client 0 的數據將固定對應到 Data Slice 0)")

    print("-" * 50)
    print(f"📂 數據集: {args.dataset}")
    print(f"🔄 總輪數: {args.total_round}")
    print("="*50 + "\n")
    # ========================================================

    # ==========================================================
    # 👇 [修正] 必須在這裡定義 output_dir，後面的程式碼才抓得到！
    # ==========================================================
    
    # 1. 建立實驗標籤 (資料夾名稱)
    experiment_tag = f"Bad{args.bad_client_num}_Rate{args.ba_poison_rate}_Shuffle{args.shuffle}"
    
    # 2. 決定根目錄 (如果沒有傳 base_dir 參數，就預設用 experiment_results_collection)
    if hasattr(args, 'base_dir'):
        base_dir = args.base_dir
    else:
        #如果您還沒在 load_argument 加入 base_dir，就用這個預設值
        base_dir = "experiment_results_collection"

    # 3. 定義 output_dir (這就是錯誤訊息說缺少的變數)
    output_dir = os.path.join(base_dir, experiment_tag)
    
    # 4. 建立資料夾
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 本次實驗輸出路徑: {output_dir}")
    # ==========================================================

    #################################### env config ####################################

    if args.device == "cpu":
        device = torch.device(f"cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    set_random_seed(args.seed)

    #################################### FLconfig ####################################
    # ==========================================================
    # 📚 [修改] 動態載入資料集與設定類別數 (統一 Resize 到 32x32)
    # ==========================================================
    client_optimizer = partial(torch.optim.SGD, lr=args.learning_rate)
    
    dataset_name = args.dataset.lower()
    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'tinyimagenet':
        num_classes = 200
    else:
        raise ValueError("❌ 不支援的資料集！請選擇 cifar10, cifar100, 或 tinyimagenet")

    print(f"📦 正在準備資料集: {dataset_name.upper()} (類別數: {num_classes})")

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 👈 關鍵防護網：強制縮放
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 👈 關鍵防護網：強制縮放
        transforms.ToTensor(),
    ])

    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
        train_dataset_labels = train_dataset.targets
        test_dataset_labels = test_dataset.targets
    elif dataset_name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=test_transform)
        train_dataset_labels = train_dataset.targets
        test_dataset_labels = test_dataset.targets
    elif dataset_name == 'tinyimagenet':
        tiny_dir = './data/tiny-imagenet-200'
        if not os.path.exists(tiny_dir):
            raise FileNotFoundError("❌ 找不到 Tiny ImageNet！請先執行 python prepare_tiny_imagenet.py")
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(tiny_dir, 'train'), transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder(os.path.join(tiny_dir, 'val'), transform=test_transform)
        train_dataset_labels = train_dataset.targets
        test_dataset_labels = test_dataset.targets
    # ==========================================================    
    
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    # train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=False, transform=transform)
    # test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=False, transform=transform)
    # train_dataset_labels = train_dataset.targets
    # test_dataset_labels = test_dataset.targets
    # num_classes = 10
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
    client_train_sample_nums = [int(len(train_dataset) / args.client_num) for _ in range(args.client_num)]
    client_test_sample_nums = [int(len(test_dataset) / args.client_num) for _ in range(args.client_num)]
    class_priors = np.random.dirichlet(alpha=[args.dir_alpha] * num_classes, size=args.client_num)
    client_train_data_indices = client_inner_dirichlet_partition(train_dataset_labels, args.client_num,
                                                                    num_classes=num_classes,
                                                                    dir_alpha=args.dir_alpha,
                                                                    client_sample_nums=client_train_sample_nums,
                                                                    class_priors=class_priors)
    client_test_data_indices = client_inner_dirichlet_partition(test_dataset_labels, args.client_num,
                                                                num_classes=num_classes,
                                                                dir_alpha=args.dir_alpha,
                                                                client_sample_nums=client_test_sample_nums,
                                                                class_priors=class_priors)
    
    
    # ==========================================
    # 👇 [新增] 將測試集的資料分佈 (Indices) 存檔！
    # ==========================================
    indices_save_path = os.path.join(output_dir, "client_test_data_indices.pt")
    torch.save(client_test_data_indices, indices_save_path)
    print(f"💾 客戶端資料分佈已儲存至: {indices_save_path}")
    # ==========================================
    
    # ==========================================
    # 直接呼叫 utils.py 的現成功能
    # ==========================================
    from utils import partition_report # 記得在最上面 import，或是寫在這裡

    print("\n📊 生成數據分佈報告 (partition_report)...")
    
    # 這會直接在終端機印出漂亮的表格，顯示每個 Client 有幾張貓、幾張狗
    partition_report(train_dataset.targets, client_train_data_indices, 
                     class_num=num_classes, verbose=True, file="data_distribution.csv")
                     
    print("✅ 詳細分佈已儲存至 data_distribution.csv")
    # ==========================================

    client_train_dataloaders = [torch.utils.data.DataLoader(train_dataset, batch_size=args.client_batch,
                                                            sampler=SubsetRandomSampler(
                                                                client_train_data_indices[i]), drop_last=True)
                                for i in range(args.client_num)]
    client_test_dataloaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.client_batch,
                                                            sampler=SubsetRandomSampler(client_test_data_indices[i]), drop_last=True)
                                for i in range(args.client_num)]

    #################################### client config ####################################

    # clients = []
    # clients = [BasicClient(get_resnet(size=10, num_classes=num_classes).to(device), client_train_dataloaders[i], client_test_dataloaders[i],
    #                         torch.nn.CrossEntropyLoss(), client_optimizer) for i in range(args.client_num - args.bad_client_num)]
    # clients.extend([PoisonClient(get_resnet(size=10, num_classes=num_classes).to(device), client_train_dataloaders[i],
    #                                 client_test_dataloaders[i], torch.nn.CrossEntropyLoss(), client_optimizer,
    #                                 poison_func=None)
    #                 for i in range(args.client_num - args.bad_client_num, args.client_num)])
    # # 👇 修改這裡：改成由參數決定要不要洗牌
    # if args.shuffle == 1:
    #     shuffle(clients)
    #     shuffle_status = "✅ 啟用 (Enabled)"
    # else:
    #     # 不洗牌，Client 0 就是原本的 Client 0
    #     shuffle_status = "❌ 停用 (Disabled) - 順序固定"

    # for idx, client in enumerate(clients):
    #     client.local_model.device = device
    #     client.cid = idx

   
    # ==========================================================
    # 🎯 [統一版 Client Config] 動態選擇演算法與均勻安插攻擊者
    # ==========================================================
        
    clients = []
    
    # 1. 根據 args.pfl 參數，動態決定要實例化哪一種 Client 類別
    pfl_algo = getattr(args, 'pfl', 'fedavg').lower()
    
    if pfl_algo == "fedprox":
        GoodClientClass = FedProxClient
        BadClientClass = PoisonFedProxClient
    elif pfl_algo == "ditto":
        GoodClientClass = DittoClient
        BadClientClass = PoisonDittoClient
    elif pfl_algo == "scaffold": 
        GoodClientClass = SCAFFOLDClient
        BadClientClass = PoisonSCAFFOLDClient
    elif pfl_algo == "fedpac":   #FedPAC 選項
        GoodClientClass = FedPACClient
        BadClientClass = PoisonFedPACClient
    else:
        # FedAvg, FedBN, FedRep 的 Client 端邏輯是一樣的，都用 BasicClient 即可
        # (FedBN 與 FedRep 的特殊處理會在 pfl.py 和 server 端進行)
        GoodClientClass = BasicClient
        BadClientClass = PoisonClient

    print(f"🔧 正在使用演算法: {pfl_algo.upper()} 初始化 Clients...")

    # 2. 計算均勻安插的間隔 (例如 100 // 20 = 5)
    interval = args.client_num // args.bad_client_num if args.bad_client_num > 0 else 99999
    bad_count = 0

    for i in range(args.client_num):
        
        # 3. 判斷當前的 i 是不是壞人 (均勻安插邏輯)
        if args.shuffle == 0 and (i + 1) % interval == 0 and bad_count < args.bad_client_num:
            is_bad = True
        elif args.shuffle == 1 and i >= (args.client_num - args.bad_client_num):
            is_bad = True
        else:
            is_bad = False

        # 4. 準備共通的初始化參數
        # client_model = get_resnet(size=args.model_size, num_classes=num_classes).to(device)
        client_model = create_model(args.model, num_classes).to(device)
        train_loader = client_train_dataloaders[i]
        test_loader = client_test_dataloaders[i]
        loss_function = torch.nn.CrossEntropyLoss()
        
        # 5. 實例化 Client 並綁定專屬 DataLoader
        # if is_bad:
        #     bad_count += 1
        #     client = BadClientClass(
        #         client_model, train_loader, test_loader, loss_function, client_optimizer,
        #         poison_func=None  # 稍後會在 FBA 模組中由 use_our_attack 掛載真正的攻擊邏輯
        #     )
        # else:
        #     client = GoodClientClass(
        #         client_model, train_loader, test_loader, loss_function, client_optimizer
        #     )
        # if is_bad:
        #     bad_count += 1
        #     #  client_lr 參數 (可以用 kwargs 動態傳遞，或直接判斷)
        #     if pfl_algo == "scaffold":
        #         client = BadClientClass(client_model, train_loader, test_loader, loss_function, client_optimizer, poison_func=None, client_lr=0.01)
        #     else:
        #         client = BadClientClass(client_model, train_loader, test_loader, loss_function, client_optimizer, poison_func=None)
        # else:
        #     if pfl_algo == "scaffold":
        #         client = GoodClientClass(client_model, train_loader, test_loader, loss_function, client_optimizer, client_lr=0.01)
        #     else:
        #         client = GoodClientClass(client_model, train_loader, test_loader, loss_function, client_optimizer)
        
        # (大約在 DEF_main.py 實例化 Client 的地方)
        if is_bad:
            bad_count += 1
            client = BadClientClass(
                client_model, train_loader, test_loader, loss_function, client_optimizer, poison_func=None
            )
        else:
            client = GoodClientClass(
                client_model, train_loader, test_loader, loss_function, client_optimizer
            )

        client.local_model.device = device
        client.cid = i
        clients.append(client)

    # 6. 洗牌與身分確認
    if args.shuffle == 1:
        from random import shuffle
        shuffle(clients)
        shuffle_status = "✅ 啟用 (Enabled)"
        # 洗牌後重新分配 cid
        for idx, client in enumerate(clients):
            client.cid = idx
    else:
        shuffle_status = f"❌ 停用 (Disabled) - 順序固定，已均勻安插 {bad_count} 個攻擊者"
        
    print(f"✅ 成功建立 {len(clients)} 個 {pfl_algo.upper()} Clients (含 {bad_count} 個攻擊者)")
    # ==========================================================

    #################################### server config ####################################

    # global_model = get_resnet(size=10, num_classes=num_classes)
    global_model = create_model(args.model, num_classes)
    
    # server = BasicServer(global_model.to(device))
    # server.global_model.device = device

    # 根據演算法選擇 Server
    if pfl_algo == "scaffold":
        server = SCAFFOLDServer(global_model.to(device), total_clients=args.client_num)
    elif pfl_algo == "fedpac": 
        server = FedPACServer(global_model.to(device), num_classes=num_classes)
    else:
        server = BasicServer(global_model.to(device))
        
    server.global_model.device = device

    server.agg_rule = args.agg_rule

    ############################### pfl config ###############################

    if args.pfl == "fedbn":
        use_fedbn(server)
    elif args.pfl == "fedrep":
        use_fedrep(server, clients, head_keyword="fc") # ResNet 的最後一層通常叫 fc

    ############################### backdoor attack config ###############################

    if args.ba == "our":
        # 👇 [修改] 將 args.use_pgd 傳入函式
        full_poison_func = use_our_attack(clients, server, args.ba_target_label, args.ba_poison_rate, args.use_pgd)
        
        # # ==========================================================
        # # 👇 [修改] 將 Generator 與存檔路徑掛載到 Server 上
        # # ==========================================================
        # if hasattr(full_poison_func, 'trigger_gen'):
        #     print("🔗 已將 Generator 掛載至 Server 以便進行過程記錄")
        #     server.trigger_gen = full_poison_func.trigger_gen
            
        #     # 設定 Generator 的過程存檔資料夾
        #     # output_dir 是我們之前定義的動態路徑 (例如 experiment_results_collection/Bad10_Rate0.2...)
        #     output_dir = 'generaTor'
        #     gen_ckpt_dir = os.path.join(output_dir, "generator_checkpoints3")
        #     os.makedirs(gen_ckpt_dir, exist_ok=True)
        #     server.gen_save_dir = gen_ckpt_dir
        # # ==========================================================
        
    
    ################################## run fl ##################################

    basic_fl_process(server, clients, local_steps=args.client_local_step, training_rounds=args.total_round,
                     select_rule=partial(random_select, nums=args.select_client_num_per_round))

    ################################## compute res ##################################

    # acc_ls, asr_ls = [], []
    # for client in clients:

    #     accuracy = evaluate_accuracy(client.local_model, client.test_dataloader)
    #     asr = evaluate_accuracy(client.local_model, client.test_dataloader, full_poison_func)
    #     print(f"Client id: {client.cid} \t Accuracy: {accuracy} \t ASR: {asr}")

    #     acc_ls.append(accuracy), asr_ls.append(asr)
    

    # print(f"Avg acc: {torch.Tensor(acc_ls).mean().item():.2f}\tAcc std: {torch.Tensor(acc_ls).std().item():.2f}\t"
    #         f"Avg ASR: {torch.Tensor(asr_ls).mean().item():.2f}\tASR std: {torch.Tensor(acc_ls).std().item():.2f}")

    print("\n📊 正在計算最終準確率 (Final Evaluation)...")

    # 定義存檔路徑 (存到該次實驗的資料夾下)
    txt_log_path = os.path.join(output_dir, "final_results.txt")
    csv_log_path = os.path.join(output_dir, "final_results.csv")

    acc_ls, asr_ls = [], []
    
    # 準備寫入檔案
    import csv
    
    with open(txt_log_path, "w", encoding="utf-8") as f_txt, \
         open(csv_log_path, "w", newline="", encoding="utf-8") as f_csv:
        
        # CSV 寫入器初始化
        writer = csv.writer(f_csv)
        writer.writerow(["Client_ID", "Type", "Accuracy", "ASR"]) # 標頭

        # 標頭寫入 TXT
        f_txt.write(f"Experiment: {experiment_tag}\n")
        f_txt.write("="*50 + "\n")

        for client in clients:
            # 判斷客戶端類型 (好人/壞人)
            c_type = "Malicious" if "Poison" in type(client).__name__ else "Benign"
            
            # 計算數據
            accuracy = evaluate_accuracy(client.local_model, client.test_dataloader)
            asr = evaluate_accuracy(client.local_model, client.test_dataloader, full_poison_func)
            
            # 1. 顯示在螢幕 (原本的功能)
            log_str = f"Client id: {client.cid} ({c_type}) \t Accuracy: {accuracy:.4f} \t ASR: {asr:.4f}"
            print(log_str)

            # 2. 寫入 TXT
            f_txt.write(log_str + "\n")

            # 3. 寫入 CSV
            writer.writerow([client.cid, c_type, accuracy, asr])

            acc_ls.append(accuracy)
            asr_ls.append(asr)
        
        # 計算統計數據
        avg_acc = torch.Tensor(acc_ls).mean().item()
        std_acc = torch.Tensor(acc_ls).std().item()
        avg_asr = torch.Tensor(asr_ls).mean().item()
        std_asr = torch.Tensor(asr_ls).std().item()

        # 準備總結字串
        summary_str = (
            f"\n" + "="*50 + "\n"
            f"Avg acc: {avg_acc:.2f}\tAcc std: {std_acc:.2f}\t"
            f"Avg ASR: {avg_asr:.2f}\tASR std: {std_asr:.2f}\n"
            f"="*50 + "\n"
        )
        
        # 1. 顯示在螢幕
        print(summary_str)
        
        # 2. 寫入 TXT
        f_txt.write(summary_str)
        
        # 3. 寫入 CSV 的最後一列 (總結)
        writer.writerow([])
        writer.writerow(["Average", "", avg_acc, avg_asr])
        writer.writerow(["Std", "", std_acc, std_asr])

    print(f"✅ 最終結果已儲存至:\n  - {txt_log_path}\n  - {csv_log_path}")

    
    # ==========================================
    
    #實驗儲存位置
    # output_dir = "experiment_resultsFedAvg/experiment_resultsAttack/experiment_results40_Shuffle"
    # FINAL_OUTPUT_DIR = "final_test_samples_FedAvg_attack/final_test_samples_results40_Shuffle"

    # os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 👇 [修改] 讓資料夾名稱動態化 (根據參數命名)
    # ==========================================

    # 建立一個包含關鍵參數的字串，例如: "Bad10_Rate0.2_Shuffle1"
    experiment_tag = f"Bad{args.bad_client_num}_Rate{args.ba_poison_rate}_Shuffle{args.shuffle}"

    # 設定根目錄
    base_dir = args.base_dir
    # base_dir = "experiment_results_collection"

    # 組合出最終路徑
    output_dir = os.path.join(base_dir, experiment_tag)
    FINAL_OUTPUT_DIR = os.path.join(output_dir, "test_samples") # 測試圖也跟著存進去

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

    print(f"📂 本次實驗結果將儲存於: {output_dir}")

    # 1. 儲存全域模型 (保持不變)
    if args.bad_client_num == 0:
        global_name = "clean_global_model.pth"
    else:
        global_name = "final_global_model.pth"
    torch.save(server.global_model.state_dict(), os.path.join(output_dir, global_name))
    print(f"✅ 全域模型已儲存至: {global_name}")


    # ==========================================
    # 👇 [修改重點] 智慧命名：自動區分好人與壞人
    # ==========================================
    
    # 設定：您想儲存前幾個 Client? (或者設大一點存全部)
    SAVE_CLIENT_COUNT = 100 
    
    print("-" * 30)
    print(f"💾 正在掃描並儲存前 {SAVE_CLIENT_COUNT} 個客戶端的模型...")

    for i in range(min(SAVE_CLIENT_COUNT, len(clients))):
        target_client = clients[i]
        
        # 🕵️‍♂️ 檢查身份：是壞人 (PoisonClient) 還是好人 (BasicClient)?
        if "Poison" in type(target_client).__name__:
            # 😈 壞人的命名格式
            prefix = "bad_attacker"
            icon = "😈"
        else:
            # 😇 好人的命名格式
            prefix = "clean_client"
            icon = "😇"
            
        # 組合檔名: 例如 bad_attacker_9.pth 或 clean_client_0.pth
        save_filename = f"{prefix}_{target_client.cid}.pth"
        save_path = os.path.join(output_dir, save_filename)
        
        torch.save(target_client.local_model.state_dict(), save_path)
        print(f"  {icon} [Client {target_client.cid}] 模型已儲存: {save_filename}")

    print("-" * 30)

    # =================================================================
    # 📸 最終測試集採樣與視覺化 (每類 N 張)
    # =================================================================
    print("\n" + "="*60)
    print("🧪 正在執行最終測試集採樣 (存 Clean / Raw / Vis / Poisoned)...")
    
    # 1. 設定參數
    SAMPLES_PER_CLASS = 2  # 您想要的 n (每類存幾張)
    # FINAL_OUTPUT_DIR = "final_test_samples_results16_Shuffle"
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    
    # 2. 取得 Trigger 生成器 (從我們剛修改的 fba.py 屬性中拿)
    # 注意：如果是乾淨模式，這個生成器就是隨機初始化的(沒訓練過)，這也是正常的
    if hasattr(full_poison_func, 'trigger_gen'):
        trigger_model = full_poison_func.trigger_gen
        trigger_model.eval()
    else:
        print("❌ 警告：無法取得 Trigger 生成器。請確認您已修改 fba.py 加入 eval_func.trigger_gen = trigger_gen")
        trigger_model = None

    # 3. 準備測試數據
    # 為了保持純淨，我們直接使用前面已經載入好且縮放為 32x32 的 test_dataset
    clean_testset = test_dataset

    # 根據目前的類別總數 (10, 100, 或 200) 動態記錄每類抓了幾張
    class_counts = {i: 0 for i in range(num_classes)}
    
    # 將類別名稱簡化為 Class_0, Class_1 ... 避免 CIFAR-100 或 TinyImageNet 報錯
    cifar_classes = [f"Class_{i}" for i in range(num_classes)]
    
    print(f"📂 輸出資料夾: {FINAL_OUTPUT_DIR}/")
    print(f"🔢 每類採樣數: {SAMPLES_PER_CLASS}")
    
    # 4. 開始遍歷與存圖
    with torch.no_grad():
        for i, (img, label) in enumerate(clean_testset):
            
            # 如果這一類已經抓滿了，就跳過
            if class_counts[label] >= SAMPLES_PER_CLASS:
                continue
                
            # 準備數據 (增加 Batch 維度)
            img_tensor = img.unsqueeze(0).to(device) # Shape: [1, 3, 32, 32]
            label_tensor = torch.tensor([label]).to(device)
            
            # --- A. 存原始乾淨圖 (Clean) ---
            class_name = cifar_classes[label]
            idx = class_counts[label] + 1
            # 檔名格式: ClassName_編號_類型.png
            base_name = f"{class_name}_{idx:02d}"
            
            torchvision.utils.save_image(img_tensor[0].cpu(), f"{FINAL_OUTPUT_DIR}/{base_name}_A_clean.png")
            
            # --- B. 計算並存 Trigger (Raw & Vis) ---
            if trigger_model is not None:
                # 根據 Bad-PFL 公式計算 Trigger
                # 公式: G(x) / 255 * 4
                raw_trigger = trigger_model(img_tensor) / 255. * 4.
                
                # B1. Raw (科學用，可能全黑)
                torchvision.utils.save_image(raw_trigger[0].cpu(), f"{FINAL_OUTPUT_DIR}/{base_name}_B1_trigger_raw.png")
                
                # B2. Vis (可視化用，拉伸對比)
                t_min, t_max = raw_trigger.min(), raw_trigger.max()
                vis_trigger = (raw_trigger - t_min) / (t_max - t_min + 1e-6)
                torchvision.utils.save_image(vis_trigger[0].cpu(), f"{FINAL_OUTPUT_DIR}/{base_name}_B2_trigger_vis.png")
                
                # --- C. 計算並存中毒圖 (Poisoned) ---
                # 注意：這裡我們手動合成 (Clean + Trigger)，不加 PGD 雜訊，
                # 這樣您比較能看清楚 Trigger 對原圖的影響。
                # 如果您堅持要加 PGD，可以直接呼叫 full_poison_func，但那樣不好控制存檔流程
                poisoned_img = torch.clamp(img_tensor + raw_trigger, 0, 1)
                torchvision.utils.save_image(poisoned_img[0].cpu(), f"{FINAL_OUTPUT_DIR}/{base_name}_C_poisoned.png")

            # 更新計數器
            class_counts[label] += 1
            
            # 檢查是否所有類別都抓滿了
            if all(c >= SAMPLES_PER_CLASS for c in class_counts.values()):
                break
                
    print("✅ 所有測試樣本已儲存完畢！")
    print("="*60)

    # 在 main.py 存檔區塊
    torch.save(trigger_model.state_dict(), os.path.join(output_dir, "trained_trigger_generator.pth"))
    print("💾 Trigger 生成器已備份！")