import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from utils import enable_mix_precision

# 不同 FL/PFL 演算法的 client 端訓練行為總集合，讓同一套 Bad-PFL attack 能在不同 federated learning 方法下跑實驗。

# ⚠️ 已移除 Label Smoothing 類別，恢復標準訓練邏輯

class BasicClient:
    def __init__(self, local_model, train_dataloader, test_dataloader, loss_func, optimizer):
        self.cid = None
        self.local_model = local_model
        self.local_model.device = torch.device("cpu")
        self.local_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.fetch = iter(self.local_dataloader)
        self.optimizer = optimizer(self.local_model.parameters())
        
        # ==========================================================
        # [修改] 恢復標準 Loss Function (由外部傳入，通常是 CrossEntropyLoss)
        # ==========================================================
        # print("ℹ️ [Info] Using Standard Loss Function (No Label Smoothing)")
        self.loss_func = loss_func 
        # ==========================================================

        self.scaler = torch.cuda.amp.GradScaler()
        self.client_info = {}
        self.training_info = {}
        self.registered_funcs = {}
        
        # 保留 Entropy 紀錄容器，這對分析很有幫助
        self.entropy_log = []

    def init_round(self):
        self.training_info.clear()
        self.entropy_log = []

    def receive_model(self, global_state_dict):
        self.local_model.load_state_dict(global_state_dict, strict=False)
        self.call_registered_func("before_local_training")
        
    def upload_model(self):
        self.upload_state_dict = self.local_model.state_dict()
        self.call_registered_func("before_upload_model")
        return self.upload_state_dict

    def local_update(self):
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        self.local_model.train()
        data = self.fetch_data()
        pred = self.forward(data)
        
        # ==========================================================
        # [修改] 計算並記錄 Entropy (不影響訓練，僅作觀察)
        # ==========================================================
        with torch.no_grad():
            # 👇 關鍵修改：將混合精度的 pred 轉為 float32，避免極小值被捨去！
            pred_fp32 = pred.float() 
            probs = torch.nn.functional.softmax(pred_fp32, dim=1)
            # 使用 1e-8 來避開 log(0)
            log_probs = torch.log(probs + 1e-8)
            batch_entropy = -(probs * log_probs).sum(dim=1).mean()
            self.entropy_log.append(batch_entropy.item())
        # ==========================================================

        loss = self.loss_computation(pred, data)
        self.backward_and_update(loss, self.optimizer)
        self.call_registered_func("after_local_update")
        torch.cuda.empty_cache()

    def local_fine_tuning(self, iter_nums):
        self.init_round()
        for _ in range(iter_nums):
            self.local_update()

    def fetch_data(self):
        try:
            data = next(self.fetch)
        except StopIteration:
            self.fetch = iter(self.local_dataloader)
            data = next(self.fetch)
        if isinstance(data, list):
            return [d.to(self.local_model.device) for d in data]
        else:
            return data.to(self.local_model.device)

    @enable_mix_precision
    def forward(self, data):
        pred = self.local_model(data[0])
        return pred

    @enable_mix_precision
    def loss_computation(self, pred, data):
        loss = self.loss_func(pred, data[1])
        return loss

    def backward_and_update(self, loss, optimizer):
        self.scaler.scale(loss).backward()
        self.call_registered_func("before_update")
        self.scaler.step(optimizer)
        self.scaler.update()

    def register_func(self, func, stage):
        if stage not in self.registered_funcs:
            self.registered_funcs[stage] = []
        self.registered_funcs[stage].append(func)

    def call_registered_func(self, stage):
        if stage in self.registered_funcs:
            for func in self.registered_funcs[stage]:
                func(self)


class PMClient(BasicClient):
    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer,
                 personalized_model=None, personalized_optimizer=None, personalized_loss_func=None):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer)

        self.personalized_model = copy.deepcopy(self.local_model) if personalized_model is None else personalized_model
        self.personalized_optimizer = optimizer(self.personalized_model.parameters()) if personalized_optimizer is None else personalized_optimizer(self.personalized_model.parameters())
        self.personalized_loss_func = loss_func if personalized_loss_func is None else personalized_loss_func
        self.update_order = "global_personalized" # or personalized_global

    def local_update(self):
        if self.update_order == "global_personalized":
            super().local_update()
  
        torch.cuda.empty_cache()
        self.personalized_optimizer.zero_grad()
        self.personalized_model.train()
        data = self.fetch_data()
        pred = self.personalized_forward(data)
        loss = self.loss_computation(pred, data)
        self.backward_and_update(loss, self.personalized_optimizer)
        torch.cuda.empty_cache()

        if self.update_order == "personalized_global":
            super().local_update()

    @enable_mix_precision
    def personalized_forward(self, data):
        pred = self.personalized_model(data[0])
        return pred


class PoisonClient(BasicClient):

    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer, poison_func):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer)
        self.poison_func = poison_func
    
    def fetch_data(self):
        return self.poison_func(*super().fetch_data())


class PMPoisonClient(PMClient):

    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer,
                 personalized_model=None, personalized_optimizer=None, personalized_loss_func=None, poison_func=None):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer,
                         personalized_model, personalized_optimizer, personalized_loss_func)
        self.poison_func = poison_func
    
    def fetch_data(self):
        return self.poison_func(*super().fetch_data())

# =======================FedProx
# ==========================================================
# 🚀 1. FedProx 相關 Client
# ==========================================================
class FedProxClient(BasicClient):
    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer, mu=0.01):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer)
        self.mu = mu
        self.global_params = None  # 👈 改變：不存整個模型，只存純粹的權重數值

    def receive_model(self, global_state_dict):
        # 接收 Server 模型
        super().receive_model(global_state_dict)
        
        # 💡 解法：只複製並凍結 Parameter 的數值 (脫離計算圖)，避免 deepcopy 報錯！
        self.global_params = [p.clone().detach() for p in self.local_model.parameters()]

    @enable_mix_precision
    def loss_computation(self, pred, data):
        # 1. 原始的分類 Loss
        original_loss = self.loss_func(pred, data[1])
        
        # 2. 計算 FedProx 的 Proximal Term
        proximal_term = 0.0
        if self.global_params is not None:
            for w, w_t in zip(self.local_model.parameters(), self.global_params):
                proximal_term += (w - w_t).norm(2) ** 2
                
        return original_loss + (self.mu / 2.0) * proximal_term

# 針對攻擊者的 FedProx
class PoisonFedProxClient(FedProxClient):
    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer, poison_func, mu=0.01):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer, mu)
        self.poison_func = poison_func
    
    def fetch_data(self):
        return self.poison_func(*super().fetch_data())

# ==========================Ditto
# ==========================================================
# 🚀 2. Ditto 相關 Client
# ==========================================================
class DittoClient(PMClient):
    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer,
                 personalized_model=None, personalized_optimizer=None, personalized_loss_func=None, lambda_ditto=0.1):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer,
                         personalized_model, personalized_optimizer, personalized_loss_func)
        self.lambda_ditto = lambda_ditto

    def local_update(self):
        # 1. 訓練 Global Model (等同於 BasicClient 的標準更新)
        super(PMClient, self).local_update() 

        # 2. 💡 解法：只提取並凍結剛訓練好的 Global Model 權重 (脫離計算圖)
        current_global_params = [p.clone().detach() for p in self.local_model.parameters()]

        # 3. 訓練 Personalized Model (加入 Ditto 的 Proximal 限制)
        torch.cuda.empty_cache()
        self.personalized_optimizer.zero_grad()
        self.personalized_model.train()
        
        data = self.fetch_data()
        pred = self.personalized_forward(data)
        
        # 計算 Ditto 專屬 Loss
        base_loss = self.personalized_loss_func(pred, data[1])
        proximal_term = 0.0
        for v, w in zip(self.personalized_model.parameters(), current_global_params):
            proximal_term += (v - w).norm(2) ** 2
            
        total_loss = base_loss + (self.lambda_ditto / 2.0) * proximal_term

        # 使用現有的 backward_and_update (但不使用原本的 loss_computation)
        self.scaler.scale(total_loss).backward()
        self.call_registered_func("before_update")
        self.scaler.step(self.personalized_optimizer)
        self.scaler.update()
        torch.cuda.empty_cache()

# 針對攻擊者的 Ditto
class PoisonDittoClient(DittoClient):
    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer,
                 personalized_model=None, personalized_optimizer=None, personalized_loss_func=None, lambda_ditto=0.1, poison_func=None):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer,
                         personalized_model, personalized_optimizer, personalized_loss_func, lambda_ditto)
        self.poison_func = poison_func

    def fetch_data(self):
        return self.poison_func(*super().fetch_data())
    
# ===================SCAFFOLD
# ==========================================================
# 🚀 3. SCAFFOLD 相關 Client (防爆改良版)
# ==========================================================
class SCAFFOLDClient(BasicClient):
    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer)
        # 本地控制變數 c_i
        self.local_c = {n: torch.zeros_like(p.data).to("cpu") for n, p in self.local_model.named_parameters()}
        self.global_c = None
        self.global_weights = None
        self.step_count = 0

    def receive_model(self, payload):
        super().receive_model(payload["weight"])
        self.global_c = payload["global_c"]
        self.global_weights = {n: p.clone().detach().to("cpu") for n, p in self.local_model.named_parameters()}
        self.step_count = 0

    @enable_mix_precision
    def loss_computation(self, pred, data):
        loss = self.loss_func(pred, data[1])
        if self.global_c is not None:
            scaffold_term = 0.0
            for name, param in self.local_model.named_parameters():
                c_diff = self.global_c[name].to(param.device) - self.local_c[name].to(param.device)
                
                # 💡 神奇技巧：(param - param.detach()) 正向時為 0 (保護 AMP 與 Loss)，反向時梯度為 c_diff！
                scaffold_term += torch.sum((param - param.detach()) * c_diff)
                
            loss += scaffold_term
        return loss

    def local_update(self):
        super().local_update()
        self.step_count += 1

    def upload_model(self):
        self.upload_state_dict = super().upload_model()
        
        # 💡 解法 2：動態取得當前的真實 Learning Rate，徹底避免數學爆炸
        current_lr = self.optimizer.param_groups[0]['lr']
        
        delta_c = {}
        for name, param in self.local_model.named_parameters():
            local_w = param.data.to("cpu")
            global_w = self.global_weights[name]
            
            if self.step_count > 0:
                # SCAFFOLD 核心公式
                c_new = self.local_c[name] - self.global_c[name] + (global_w - local_w) / (self.step_count * current_lr)
            else:
                c_new = self.local_c[name]
                
            delta_c[name] = c_new - self.local_c[name]
            self.local_c[name] = c_new
            
        return {"weight": self.upload_state_dict, "delta_c": delta_c}

class PoisonSCAFFOLDClient(SCAFFOLDClient):
    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer, poison_func):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer)
        self.poison_func = poison_func
    
    def fetch_data(self):
        return self.poison_func(*super().fetch_data())

# ===================fedpac
# ==========================================================
# 🚀 4. FedPAC 相關 Client (Hook 攔截特徵版)
# ==========================================================
class FedPACClient(BasicClient):
    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer, lamda=0.1, num_classes=10):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer)
        self.lamda = lamda
        self.num_classes = num_classes
        self.global_centers = None
        
        # 💡 神技：使用 Hook 自動攔截最後一層 (FC層) 的輸入，也就是我們需要的「特徵」！
        self.captured_features = None
        last_layer = list(self.local_model.children())[-1]
        last_layer.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module, input):
        self.captured_features = input[0] # 攔截特徵

    def receive_model(self, payload):
        super().receive_model(payload["weight"])
        self.global_centers = payload["global_centers"]

    @enable_mix_precision
    def loss_computation(self, pred, data):
        loss = self.loss_func(pred, data[1])
        labels = data[1]
        
        # 加入 FedPAC 特有的 Feature Alignment Loss
        if self.global_centers is not None and self.captured_features is not None:
            reg_loss = 0.0
            valid_samples = 0
            for i, label in enumerate(labels):
                y = label.item()
                if y in self.global_centers:
                    target_center = self.global_centers[y].to(self.captured_features.device)
                    # 計算 L2 距離 (拉近本地特徵與全域中心)
                    reg_loss += torch.sum((self.captured_features[i] - target_center) ** 2)
                    valid_samples += 1
            if valid_samples > 0:
                loss += (self.lamda / 2.0) * (reg_loss / valid_samples)
                
        return loss

    def upload_model(self):
        self.upload_state_dict = super().upload_model()
        
        # 訓練結束後，跑一次本地資料來計算 Local Feature Centers
        self.local_model.eval()
        local_centers = {}
        local_counts = {}
        
        # 💡 解法：把 with torch.no_grad(): 移到迴圈內部，釋放 PGD 的算力
        for raw_data in self.local_dataloader:
            images_dev = raw_data[0].to(self.local_model.device)
            labels_dev = raw_data[1].to(self.local_model.device)
            
            # 1. 若是攻擊者，要套用毒化函數 (這裡面的 PGD 需要算梯度，所以不能被 no_grad 包住)
            if hasattr(self, 'poison_func') and self.poison_func is not None:
                data = self.poison_func(images_dev, labels_dev)
                images, labels = data[0], data[1]
            else:
                images, labels = images_dev, labels_dev
            
            # 2. 模型推論抓特徵的地方才需要 no_grad 來節省顯存
            with torch.no_grad():
                _ = self.local_model(images) # 觸發 Hook 抓取特徵
                features = self.captured_features.cpu()
            
            for i, label in enumerate(labels):
                y = label.item()
                if y not in local_centers:
                    local_centers[y] = torch.zeros_like(features[i])
                    local_counts[y] = 0
                local_centers[y] += features[i]
                local_counts[y] += 1
                
        # 平均計算出真正的中心點
        for y in local_centers.keys():
            local_centers[y] /= local_counts[y]
            
        return {"weight": self.upload_state_dict, "centers": local_centers, "counts": local_counts}

class PoisonFedPACClient(FedPACClient):
    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer, poison_func, lamda=0.1, num_classes=10):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer, lamda, num_classes)
        self.poison_func = poison_func
    
    def fetch_data(self):
        return self.poison_func(*super().fetch_data())

# ===================
