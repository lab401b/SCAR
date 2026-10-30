import torch


def agg_avg(state_dicts):
    average_dict = state_dicts[0]
    for key in average_dict.keys():
        for idx in range(1, len(state_dicts)):
            average_dict[key] = average_dict[key] + state_dicts[idx][key]
        average_dict[key] = average_dict[key] / len(state_dicts)
    return average_dict


class BasicServer:
    def __init__(self, global_model):
        self.global_model = global_model
        self.agg_rule = "avg"
        self.server_info = {}
        self.training_info = {}
        self.registered_funcs = {}
        self.update = None
        self.distribute_dict = None

    @torch.no_grad()
    def distribute_model(self):
        self.distribute_dict = self.global_model.state_dict()
        self.call_registered_func("before_distribute_global")
        return self.distribute_dict

    @torch.no_grad()
    def agg_and_update(self, state_dicts):
        self.update = state_dicts
        self.update = agg_avg(state_dicts)
        self.call_registered_func("before_update_global")
        self.global_model.load_state_dict(self.update, strict=False)

    def register_func(self, func, stage):
        if stage not in self.registered_funcs:
            self.registered_funcs[stage] = []
        self.registered_funcs[stage].append(func)

    def call_registered_func(self, stage):
        if stage in self.registered_funcs:
            for func in self.registered_funcs[stage]:
                func(self)

# ==========================================================
# SCAFFOLD 專用 Server
# ==========================================================
class SCAFFOLDServer(BasicServer):
    def __init__(self, global_model, total_clients=100):
        super().__init__(global_model)
        self.total_clients = total_clients
        # 建立全域控制變數 (Global Control Variate 'c')
        self.global_c = {n: torch.zeros_like(p.data).to("cpu") for n, p in self.global_model.named_parameters()}

    @torch.no_grad()
    def distribute_model(self):
        # Server 下發模型時，同時打包「模型權重」與「全域控制變數」
        self.distribute_dict = self.global_model.state_dict()
        self.call_registered_func("before_distribute_global")
        return {"weight": self.distribute_dict, "global_c": self.global_c}

    @torch.no_grad()
    def agg_and_update(self, client_uploads):
        # client_uploads 是一個 List，裡面包含了每個 Client 上傳的 dict
        weights = [u["weight"] for u in client_uploads]
        delta_cs = [u["delta_c"] for u in client_uploads]

        # 1. 聚合並更新全域模型權重 (沿用原本的 agg_avg)
        self.update = agg_avg(weights)
        self.call_registered_func("before_update_global")
        self.global_model.load_state_dict(self.update, strict=False)

        # 2. 聚合 Client 傳上來的 Delta c，並更新全域控制變數 c
        avg_delta_c = agg_avg(delta_cs)
        ratio = len(client_uploads) / self.total_clients  # 通常是 10 / 100 = 0.1
        for name in self.global_c.keys():
            self.global_c[name] += avg_delta_c[name] * ratio


# ==========================================================
# FedPAC 專用 Server
# ==========================================================
class FedPACServer(BasicServer):
    def __init__(self, global_model, num_classes=10):
        super().__init__(global_model)
        self.num_classes = num_classes
        self.global_centers = None

    @torch.no_grad()
    def distribute_model(self):
        self.distribute_dict = self.global_model.state_dict()
        self.call_registered_func("before_distribute_global")
        # 下發權重 + 全域特徵中心
        return {"weight": self.distribute_dict, "global_centers": self.global_centers}

    @torch.no_grad()
    def agg_and_update(self, client_uploads):
        # 1. 聚合模型權重
        weights = [u["weight"] for u in client_uploads]
        self.update = agg_avg(weights)
        self.call_registered_func("before_update_global")
        self.global_model.load_state_dict(self.update, strict=False)

        # 2. 聚合特徵中心 (Feature Centers)
        new_global_centers = {}
        total_counts = {y: 0 for y in range(self.num_classes)}
        
        # 加總所有 Client 回傳的特徵與數量
        for u in client_uploads:
            for y, count in u["counts"].items():
                total_counts[y] += count
                if y not in new_global_centers:
                    new_global_centers[y] = torch.zeros_like(u["centers"][y])
                new_global_centers[y] += u["centers"][y] * count
                
        # 計算平均值
        for y in new_global_centers.keys():
            if total_counts[y] > 0:
                new_global_centers[y] /= total_counts[y]
                
        self.global_centers = new_global_centers
