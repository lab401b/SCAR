import torch
import torch.nn as nn

# def use_fedbn(server):

#     def fedbn_update(server):
#         delete_keys = []
#         for key in server.update.keys():
#             if "bn" in key or "shortcut.1" in key:
#                 delete_keys.append(key)
        
#         for key in delete_keys:
#             server.update.pop(key)

#     def fedbn_distribute(server):
#         delete_keys = []
#         for key in server.distribute_dict.keys():
#             if "bn" in key or "shortcut.1" in key:
#                 delete_keys.append(key)
        
#         for key in delete_keys:
#             server.distribute_dict.pop(key)

#     server.register_func(fedbn_update, "before_update_global")
#     server.register_func(fedbn_distribute, "before_distribute_global")

def use_fedbn(server):
    print("🔧 [PFL] 啟用動態 FedBN: 正在掃描並攔截所有的 BatchNorm 層...")
    
    # 1. 動態抓取全域模型中「真正屬於 BN 層」的變數名稱前綴
    bn_prefixes = []
    for name, module in server.global_model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_prefixes.append(name)
            
    # 2. 判斷該 key 是否屬於 BN 層的參數 (weight, bias, running_mean, running_var, num_batches_tracked)
    def is_bn_key(key):
        for prefix in bn_prefixes:
            if key.startswith(prefix + "."):
                return True
        return False

    def fedbn_update(server):
        delete_keys = [key for key in server.update.keys() if is_bn_key(key)]
        for key in delete_keys:
            server.update.pop(key)

    def fedbn_distribute(server):
        delete_keys = [key for key in server.distribute_dict.keys() if is_bn_key(key)]
        for key in delete_keys:
            server.distribute_dict.pop(key)

    server.register_func(fedbn_update, "before_update_global")
    server.register_func(fedbn_distribute, "before_distribute_global")

# =================fedrep

def use_fedrep(server, clients, head_keyword="fc"):
    print(f"🔧 [PFL] 啟用 FedRep: Server 與 Client 之間將不再傳遞 '{head_keyword}' 分類層")

    # 1. Server 下發模型時，剔除分類頭
    original_distribute = server.distribute_model
    def rep_distribute_model():
        global_state_dict = original_distribute()
        filtered_dict = {k: v for k, v in global_state_dict.items() if head_keyword not in k}
        return filtered_dict
    server.distribute_model = rep_distribute_model

    # 2. Client 上傳模型時，剔除分類頭
    def before_upload_model(client):
        for name in list(client.upload_state_dict.keys()):
            if head_keyword in name:
                del client.upload_state_dict[name]

    for client in clients:
        client.register_func(before_upload_model, "before_upload_model")
