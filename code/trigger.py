import torch
import torchvision
import os

@torch.no_grad()
def grid_trigger_adder(data, label, target_label, poison_ratio, trigger_val=1., position="left_top", trigger_size=3,
                        strategy="paste", blend_ratio=0.5):
    
    # 1. 加入這行 debug 訊息，確認函式有沒有被呼叫
    # print("DEBUG: grid_trigger_adder 被呼叫了！") 

    batch_size, channel, height, width = data.size(0), data.size(1), data.size(2), data.size(3)
    trigger_size = [trigger_size for _ in range(2)] if isinstance(trigger_size, int) else trigger_size

    poison_mask = torch.rand(batch_size, device=label.device) <= poison_ratio

    # ==================================================
    # 👇 [修改重點] 強制執行區 👇
    # ==================================================
    # 如果還沒存過圖，我們就「強迫」把第一張圖標記為中毒
    # 這樣可以繞過 poison_mask.sum() == 0 的檢查，確保一定會進入 else 區塊存圖
    if not hasattr(grid_trigger_adder, "has_saved_image"):
        poison_mask[0] = True 
    # ==================================================

    if poison_mask.sum().item() == 0:
        return data, label
    else:
        # ... (這裡保留原本產生中毒資料的程式碼) ...
        poison_data, poison_label = data.clone(), torch.full([batch_size,], target_label, device=label.device)

        if position == "left_top":
            start_height, start_width = 0, 0
        elif position == "random":
            start_height, start_width = torch.randint(0, height - trigger_size[0] + 1, (1,)).item(), torch.randint(0, width - trigger_size[1] + 1, (1,)).item()
        elif position == "center":
            start_height, start_width = height // 2 - trigger_size[0] // 2,  width // 2 - trigger_size[1] // 2
            
        if isinstance(trigger_val, float):
            trigger = torch.full((batch_size, channel, *trigger_size), trigger_val, device=label.device)
        elif isinstance(trigger_val, list):
            trigger = torch.Tensor(trigger_val).view(1, channel, *trigger_size).to(label.device).repeat(batch_size, 1, 1, 1)
        elif isinstance(trigger_val, torch.Tensor):
            trigger = trigger_val.to(label.device)

        if strategy == "paste":
            poison_data[:,:,start_height:start_height+trigger_size[0],start_width:start_width+trigger_size[1]] = trigger
        elif strategy == "blend":
            poison_data = (1 - blend_ratio) * poison_data + blend_ratio * trigger
        else:
            raise ValueError("strategy must be either 'paste' or 'blend'")

        poison_data = poison_mask.view(-1, 1, 1, 1).float() * poison_data + (~poison_mask.view(-1, 1, 1, 1)).float() * data
        poison_label = poison_mask.float() * poison_label + (~poison_mask).float() * label

        print(f"test\n")

        # ==================================================
        # 👇 存圖區 (您原本寫的) 👇
        # ==================================================
        if not hasattr(grid_trigger_adder, "has_saved_image"):
            try:
                output_dir = "experiment_results4"
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, "debug_poisoned_sample.png")
                
                # 這裡要小心，poison_data 可能在 GPU 上，要先轉 CPU
                img_to_save = poison_data[0].cpu()
                torchvision.utils.save_image(img_to_save, save_path)
                
                print(f"\n [成功] 中毒圖片已強制儲存至: {save_path}")
                grid_trigger_adder.has_saved_image = True
            except Exception as e:
                print(f" 存圖失敗: {e}")
        # ==================================================

        return poison_data, poison_label.to(torch.long)