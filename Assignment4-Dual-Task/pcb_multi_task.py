import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 設定 Matplotlib 後端以支援非互動式環境
import matplotlib
matplotlib.use('Agg')

# --- 1. 資料集定義 (Dataset Definitions) ---

class DeepPCBDetDataset(Dataset):
    """DeepPCB 目標偵測資料集 (Task A)"""
    def __init__(self, root, split='train', size=(256, 256)):
        self.img_dir = os.path.join(root, split, "images")
        self.lbl_dir = os.path.join(root, split, "labels")
        self.size = size
        if not os.path.exists(self.img_dir): raise FileNotFoundError(f"找不到 DeepPCB: {self.img_dir}")
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.base_transform = T.Compose([T.ToPILImage(), T.Resize(size), T.ToTensor()])
        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self): return len(self.img_files)
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        h_orig, w_orig = img.shape[:2] if img is not None else (640, 640)
        img_t = self.base_transform(img) if img is not None else torch.zeros((3, self.size[0], self.size[1]))
        lbl_path = os.path.join(self.lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    p = [float(x) for x in line.split()]
                    if len(p) == 5:
                        x1, y1, x2, y2, cls = p
                        xc, yc = (x1 + x2) / 2 / w_orig, (y1 + y2) / 2 / h_orig
                        bw, bh = (x2 - x1) / w_orig, (y2 - y1) / h_orig
                        boxes.append([cls, xc, yc, bw, bh])
        return self.norm(img_t), torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5)), img_t

def det_collate_fn(batch):
    imgs_norm, targets, imgs_raw = zip(*batch)
    return torch.stack(imgs_norm, 0), list(targets), torch.stack(imgs_raw, 0)

class VisASegDataset(Dataset):
    """VisA PCB 語意分割資料集 (Task B)"""
    def __init__(self, root, pcb_folders=['pcb1', 'pcb2', 'pcb3', 'pcb4'], size=(256, 256)):
        self.root, self.size, self.samples = root, size, []
        for folder in pcb_folders:
            csv_path = os.path.join(root, folder, "image_anno.csv")
            if not os.path.exists(csv_path): continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.samples.append({
                    'img': os.path.join(root, row['image']),
                    'mask': os.path.join(root, row['mask']) if pd.notna(row['mask']) else None,
                    'label': row['label']
                })
        self.img_transform = T.Compose([T.ToPILImage(), T.Resize(size), T.ToTensor()])
        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.mask_transform = T.Compose([T.ToPILImage(), T.Resize(size, interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s['img'])
        
        # 強制將 Mask 轉為 0/255 二值化圖像，避免 ToTensor 後產生極小數值
        if s['mask'] and os.path.exists(s['mask']):
            mask = cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        else:
            mask = np.zeros((self.size[0], self.size[1]), dtype=np.uint8)
            
        img_t = self.img_transform(img)
        return self.norm(img_t), self.mask_transform(mask), img_t

# --- 2. 雙任務架構 (ResNet18-UNet) ---

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MultiTaskResUNet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.l1, self.l2, self.l3, self.l4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        self.d1 = DecoderBlock(512, 256, 256)
        self.d2 = DecoderBlock(256, 128, 128)
        self.d3 = DecoderBlock(128, 64, 64)
        self.d4 = DecoderBlock(64, 64, 64)
        self.final_seg = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.Conv2d(32, 1, 1)
        )
        self.det_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 1, 1) 
        )

    def forward(self, x, task='both'):
        s0 = self.stem(x); s0_p = self.maxpool(s0)
        s1 = self.l1(s0_p); s2 = self.l2(s1); s3 = self.l3(s2); s4 = self.l4(s3)
        out = {}
        if task in ['det', 'both']: out['det'] = self.det_head(s4)
        if task in ['seg', 'both']:
            x_seg = self.d1(s4, s3); x_seg = self.d2(x_seg, s2)
            x_seg = self.d3(x_seg, s1); x_seg = self.d4(x_seg, s0)
            out['seg'] = self.final_seg(x_seg)
        return out

# --- 3. 優化後的損失函數與評估 ---

class FocalDiceLoss(nn.Module):
    """專為極端不平衡設計的混合損失：Focal Loss + Instance-level Dice Loss"""
    def __init__(self, alpha=0.8, gamma=2.0, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        # 1. Focal Loss (Pixel-wise)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = (alpha_t * (1 - p_t) ** self.gamma * bce).mean()

        # 2. Instance-level Dice Loss
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        
        return focal_loss + dice_loss.mean()

def evaluate(model, det_loader, seg_loader, device):
    model.eval()
    det_hits, det_total = 0, 1e-7
    normal_iou_list, anomaly_iou_list = [], []
    max_p, pixels_03, pixels_01 = 0.0, 0, 0
    with torch.no_grad():
        for imgs, bboxes, _ in det_loader:
            imgs = imgs.to(device)
            out = torch.sigmoid(model(imgs, task='det')['det'][:, 0, :, :])
            for b in range(imgs.size(0)):
                if len(bboxes[b]) == 0: continue
                idx = out[b].view(-1).argmax().item()
                py, px = (idx // 8) / 8.0, (idx % 8) / 8.0
                hit = any(abs(px + 0.06 - box[1]) < (box[3]/2 + 0.05) and abs(py + 0.06 - box[2]) < (box[4]/2 + 0.05) for box in bboxes[b])
                if hit: det_hits += 1
                det_total += 1
                
        for imgs, masks, _ in seg_loader:
            imgs = imgs.to(device)
            masks = (masks.to(device) > 0.5).float()
            probs = torch.sigmoid(model(imgs, task='seg')['seg'])
            max_p = max(max_p, probs.max().item())
            
            preds_05 = (probs > 0.5).float()
            preds_03 = (probs > 0.3).float()
            preds_01 = (probs > 0.1).float()
            pixels_03 += preds_03.sum().item()
            pixels_01 += preds_01.sum().item()
            
            inter = (preds_05 * masks).sum(dim=(1,2,3))
            uni = (preds_05 + masks).clamp(0,1).sum(dim=(1,2,3))
            
            for b in range(imgs.size(0)):
                is_anomaly = masks[b].sum() > 0
                if uni[b] == 0:
                    normal_iou_list.append(1.0)
                else:
                    iou_val = (inter[b] / uni[b]).item()
                    if is_anomaly:
                        anomaly_iou_list.append(iou_val)
                    else:
                        normal_iou_list.append(iou_val)
                
    return det_hits/det_total, np.mean(normal_iou_list) if normal_iou_list else 0.0, np.mean(anomaly_iou_list) if anomaly_iou_list else 0.0, max_p, pixels_03, pixels_01

# --- 4. 可視化與統計摘要 ---

def visualize_diagnostics(dataset_det, dataset_seg, tr_idx, save_prefix="gallery"):
    print("正在產出資料集診斷圖表...")
    
    # --- DeepPCB Gallery ---
    class_names = {1:"Open", 2:"Short", 3:"Mousebite", 4:"Spur", 5:"Copper", 6:"Pin-hole"}
    found_det = {}
    for i in range(len(dataset_det)):
        _, targets, img_raw = dataset_det[i]
        for box in targets:
            c = int(box[0])
            if c in class_names and c not in found_det: found_det[c] = (img_raw, targets)
        if len(found_det) == 6: break
        
    plt.figure(figsize=(15, 8))
    for i, (cls, (img, boxes)) in enumerate(sorted(found_det.items())):
        ax = plt.subplot(2, 3, i+1); ax.imshow(img.permute(1,2,0).numpy()); h, w = 256, 256
        for box in boxes:
            if int(box[0]) == cls:
                x1, y1 = (box[1]-box[3]/2)*w, (box[2]-box[4]/2)*h
                ax.add_patch(plt.Rectangle((x1,y1), box[3]*w, box[4]*h, fill=False, edgecolor='cyan', linewidth=2))
        ax.set_title(f"DeepPCB: {class_names[cls]}"); ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_deeppcb.png")
    
    # --- VisA Gallery 修復：尋找並繪製獨特的瑕疵類別 ---
    found_visa = {}
    for idx in tr_idx:
        s = dataset_seg.samples[idx]
        # 確保存在 mask 且為瑕疵樣本，提取其 label
        if pd.notna(s['mask']):
            lbl = str(s.get('label', 'defect'))
            if lbl not in found_visa:
                found_visa[lbl] = idx
        # 限制最多顯示 6 種相異瑕疵類別以維持版面配置
        if len(found_visa) >= 6: 
            break

    n_visa = max(1, len(found_visa))
    plt.figure(figsize=(max(12, 3 * n_visa), 6))
    for i, (lbl, idx) in enumerate(found_visa.items()):
        _, mask, img_raw = dataset_seg[idx]
        ax1 = plt.subplot(2, n_visa, i+1); ax1.imshow(img_raw.permute(1,2,0).numpy()); ax1.axis('off')
        ax1.set_title(f"VisA: {lbl}")
        
        ax2 = plt.subplot(2, n_visa, i+1+n_visa); ax2.imshow(mask[0].numpy(), cmap='magma'); ax2.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_visa.png")

def plot_learning_curves(history):
    epochs = range(1, len(history['det_acc']) + 1)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1); plt.plot(epochs, history['det_acc'], 'b-'); plt.title('Val Det Accuracy'); plt.grid(True)
    plt.subplot(1, 3, 2); plt.plot(epochs, history['seg_iou_norm'], 'g-'); plt.title('Val Seg IoU (Normal)'); plt.grid(True)
    plt.subplot(1, 3, 3); plt.plot(epochs, history['seg_iou_anom'], 'r-'); plt.title('Val Seg IoU (Anomaly)'); plt.grid(True)
    plt.savefig("learning_curves.png")

def visualize_final_predictions(model, det_loader, seg_loader, device, prefix="test"):
    model.eval()
    class_names = {1:"Open", 2:"Short", 3:"Mousebite", 4:"Spur", 5:"Copper", 6:"Pin-hole"}
    
    plt.figure(figsize=(15, 5))
    imgs, bboxes, raw_imgs = next(iter(det_loader))
    with torch.no_grad(): out = torch.sigmoid(model(imgs.to(device), task='det')['det'][:, 0, :, :])
    for i in range(min(4, len(raw_imgs))):
        ax = plt.subplot(1, 4, i+1); img_np = raw_imgs[i].permute(1,2,0).numpy(); ax.imshow(img_np)
        for box in bboxes[i]:
            cls_id = int(box[0])
            cls_name = class_names.get(cls_id, str(cls_id))
            x1, y1 = (box[1]-box[3]/2)*256, (box[2]-box[4]/2)*256
            ax.add_patch(plt.Rectangle((x1,y1), box[3]*256, box[4]*256, fill=False, edgecolor='blue', linewidth=2))
            # 加上白底藍字的類別標籤，提高可讀性
            ax.text(x1, y1 - 2, cls_name, color='blue', fontsize=9, weight='bold', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
        idx = out[i].view(-1).argmax().item()
        ax.plot((idx % 8)*32+16, (idx // 8)*32+16, 'ro'); ax.axis('off')
    plt.savefig(f"{prefix}_det_pred.png")
    
    plt.figure(figsize=(12, 8))
    imgs_s, masks_s, raw_imgs_s = next(iter(seg_loader))
    with torch.no_grad(): probs = torch.sigmoid(model(imgs_s.to(device), task='seg')['seg']).cpu().numpy()
    for i in range(min(3, len(raw_imgs_s))):
        ax1 = plt.subplot(3, 3, 3*i+1); ax1.imshow(raw_imgs_s[i].permute(1,2,0).numpy()); ax1.axis('off')
        ax2 = plt.subplot(3, 3, 3*i+2); ax2.imshow(masks_s[i,0], cmap='gray'); ax2.axis('off')
        ax3 = plt.subplot(3, 3, 3*i+3); ax3.imshow(probs[i,0] > 0.5, cmap='viridis'); ax3.axis('off')
    plt.savefig(f"{prefix}_seg_pred.png")

def redraw_defect_visualizations(model, dataset_det, dataset_seg, device, prefix="redraw"):
    """
    獨立繪製包含「真實瑕疵」的視覺化圖表。
    強制篩選出 Ground Truth 確實包含瑕疵的樣本進行繪製，避免隨機抽到全黑的良品。
    此函式可在模型訓練完成後，或於互動式環境中獨立呼叫，無需重新訓練。
    """
    print("正在重新繪製瑕疵樣本視覺化...")
    class_names = {1:"Open", 2:"Short", 3:"Mousebite", 4:"Spur", 5:"Copper", 6:"Pin-hole"}
    
    # 1. 篩選 VisA 中真正有瑕疵的樣本 (mask 路徑存在)
    defect_indices = [i for i, s in enumerate(dataset_seg.samples) if pd.notna(s['mask'])]
    np.random.seed(42)  # 固定亂數種子方便重現
    selected_visa_idx = np.random.choice(defect_indices, min(5, len(defect_indices)), replace=False)
    
    # --- 繪製 VisA Ground Truth 診斷圖 ---
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(selected_visa_idx):
        _, mask, img_raw = dataset_seg[idx]
        ax1 = plt.subplot(2, 5, i+1); ax1.imshow(img_raw.permute(1,2,0).numpy()); ax1.axis('off')
        ax2 = plt.subplot(2, 5, i+6); ax2.imshow(mask[0].numpy(), cmap='magma'); ax2.axis('off')
        if i == 2: ax1.set_title("VisA Defect Images")
        if i == 2: ax2.set_title("VisA Ground Truth Masks")
    plt.tight_layout()
    plt.savefig(f"{prefix}_visa_gt_only.png")
    
    # --- 繪製 VisA 分割預測圖 ---
    if model is not None:
        model.eval()
        plt.figure(figsize=(12, 8))
        subset = Subset(dataset_seg, selected_visa_idx[:3])
        loader = DataLoader(subset, batch_size=3, shuffle=False)
        imgs_s, masks_s, raw_imgs_s = next(iter(loader))
        
        with torch.no_grad():
            probs = torch.sigmoid(model(imgs_s.to(device), task='seg')['seg']).cpu().numpy()
            
        for i in range(min(3, len(raw_imgs_s))):
            ax1 = plt.subplot(3, 3, 3*i+1); ax1.imshow(raw_imgs_s[i].permute(1,2,0).numpy()); ax1.axis('off')
            ax2 = plt.subplot(3, 3, 3*i+2); ax2.imshow(masks_s[i,0].numpy(), cmap='gray'); ax2.axis('off')
            ax3 = plt.subplot(3, 3, 3*i+3); ax3.imshow(probs[i,0] > 0.5, cmap='viridis'); ax3.axis('off')
            if i == 0:
                ax1.set_title("Image")
                ax2.set_title("Ground Truth")
                ax3.set_title("Prediction (>0.5)")
        plt.tight_layout()
        plt.savefig(f"{prefix}_seg_pred_defects.png")
        
    # --- 繪製 DeepPCB 偵測預測圖 ---
    if model is not None and dataset_det is not None:
        model.eval()
        plt.figure(figsize=(15, 5))
        dl_det = DataLoader(dataset_det, batch_size=4, shuffle=True, collate_fn=det_collate_fn)
        imgs, bboxes, raw_imgs = next(iter(dl_det))
        
        with torch.no_grad(): 
            out = torch.sigmoid(model(imgs.to(device), task='det')['det'][:, 0, :, :])
            
        for i in range(min(4, len(raw_imgs))):
            ax = plt.subplot(1, 4, i+1); img_np = raw_imgs[i].permute(1,2,0).numpy(); ax.imshow(img_np)
            for box in bboxes[i]:
                cls_id = int(box[0])
                cls_name = class_names.get(cls_id, str(cls_id))
                x1, y1 = (box[1]-box[3]/2)*256, (box[2]-box[4]/2)*256
                ax.add_patch(plt.Rectangle((x1,y1), box[3]*256, box[4]*256, fill=False, edgecolor='blue', linewidth=2))
                # 加上白底藍字的類別標籤
                ax.text(x1, y1 - 2, cls_name, color='blue', fontsize=9, weight='bold', 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
            idx = out[i].view(-1).argmax().item()
            ax.plot((idx % 8)*32+16, (idx // 8)*32+16, 'ro'); ax.axis('off')
            if i == 0: ax.set_title("DeepPCB Detection")
        plt.tight_layout()
        plt.savefig(f"{prefix}_det_pred_defects.png")
    print(f"重新繪製完成，檔案已儲存 ({prefix}_*.png)。")

# --- 5. 訓練流程 ---

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # 1. 數據統計與拆分
        train_det = DeepPCBDetDataset("../DeepPCB", 'train')
        valid_det = DeepPCBDetDataset("../DeepPCB", 'valid')
        test_det  = DeepPCBDetDataset("../DeepPCB", 'test')
        full_seg  = VisASegDataset("../VisA_PCB_Subset")
        
        labels = np.array([s['label'] for s in full_seg.samples]); l_counts = pd.Series(labels).value_counts()
        strat = np.array([l if l_counts[l] >= 3 else "rare" for l in labels])
        tr_val_idx, te_idx = train_test_split(np.arange(len(full_seg)), test_size=0.1, stratify=strat, random_state=42)
        tr_idx, va_idx = train_test_split(tr_val_idx, test_size=0.11, stratify=strat[tr_val_idx], random_state=42)
        
        print("\n" + "="*55)
        print("數據集樣本統計 (Dataset Statistics):")
        print(f"Task A (DeepPCB 偵測): Train={len(train_det)}, Val={len(valid_det)}, Test={len(test_det)}")
        print(f"Task B (VisA 分割):     Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")
        print("="*55 + "\n")
        
        visualize_diagnostics(train_det, full_seg, tr_idx)

        dl_tr_d = DataLoader(train_det, batch_size=16, shuffle=True, collate_fn=det_collate_fn)
        dl_va_d = DataLoader(valid_det, batch_size=16, collate_fn=det_collate_fn)
        dl_te_d = DataLoader(test_det, batch_size=16, collate_fn=det_collate_fn)
        
        # --- Plan C: Oversampling 強制採樣平衡 ---
        train_seg_subset = Subset(full_seg, tr_idx)
        has_defect = np.array([1 if full_seg.samples[i]['mask'] is not None else 0 for i in tr_idx])
        class_counts = np.bincount(has_defect)
        weights = np.zeros_like(has_defect, dtype=np.float32)
        if len(class_counts) > 0 and class_counts[0] > 0: weights[has_defect == 0] = 1.0 / class_counts[0]
        if len(class_counts) > 1 and class_counts[1] > 0: weights[has_defect == 1] = 1.0 / class_counts[1]
        sampler_s = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        
        dl_tr_s = DataLoader(train_seg_subset, batch_size=16, sampler=sampler_s)
        dl_va_s = DataLoader(Subset(full_seg, va_idx), batch_size=16)
        dl_te_s = DataLoader(Subset(full_seg, te_idx), batch_size=16)

        # 2. 模型初始化與優化器
        model = MultiTaskResUNet18().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # 使用新的 FocalDiceLoss，針對 PCB 瑕疵設定高 alpha 權重
        seg_loss_fn = FocalDiceLoss(alpha=0.85, gamma=2.0).to(device)

        print(f"--- 雙階段多任務監控訓練啟動 ---")
        history = {'det_acc': [], 'seg_iou_norm': [], 'seg_iou_anom': []}
        SEG_ONLY, TOTAL = 10, 25

        for epoch in range(TOTAL):
            model.train()
            d_iter, s_iter = iter(dl_tr_d), iter(dl_tr_s)
            steps = max(len(dl_tr_d), len(dl_tr_s))
            task_mode = 'seg' if epoch < SEG_ONLY else 'both'
            
            # --- Plan A: 隔離單任務預熱期的梯度 (動態凍結 Backbone) ---
            is_joint = (task_mode == 'both')
            for param in model.stem.parameters(): param.requires_grad = is_joint
            for param in model.l1.parameters(): param.requires_grad = is_joint
            for param in model.l2.parameters(): param.requires_grad = is_joint
            for param in model.l3.parameters(): param.requires_grad = is_joint
            for param in model.l4.parameters(): param.requires_grad = is_joint
            
            for _ in range(steps):
                optimizer.zero_grad(); loss = 0
                
                # 完整解包 Iterator 避免 StopIteration 例外處理中的變數遺失
                try: 
                    imgs_s, masks_s, _ = next(s_iter)
                except StopIteration: 
                    s_iter = iter(dl_tr_s)
                    imgs_s, masks_s, _ = next(s_iter)
                
                loss += seg_loss_fn(model(imgs_s.to(device), task='seg')['seg'], masks_s.to(device))
                
                if task_mode == 'both':
                    try: 
                        imgs_d, boxes_d, _ = next(d_iter)
                    except StopIteration: 
                        d_iter = iter(dl_tr_d)
                        imgs_d, boxes_d, _ = next(d_iter)
                        
                    targets_d = torch.zeros((imgs_d.size(0), 1, 8, 8)).to(device)
                    for b in range(len(boxes_d)):
                        for box in boxes_d[b]: targets_d[b,0,min(7,int(box[2]*8)),min(7,int(box[1]*8))] = 1.0
                    loss += nn.BCEWithLogitsLoss()(model(imgs_d.to(device), task='det')['det'], targets_d)
                    
                loss.backward(); optimizer.step()

            # 評估與紀錄
            a_d, i_s_n, i_s_a, max_p, px03, px01 = evaluate(model, dl_va_d, dl_va_s, device)
            history['det_acc'].append(a_d)
            history['seg_iou_norm'].append(i_s_n)
            history['seg_iou_anom'].append(i_s_a)
            mode_str = "[SEG-ONLY]" if epoch < SEG_ONLY else "[JOINT]"
            print(f"Epoch {epoch+1:02d} {mode_str} | Det Acc: {a_d:.6f} | Seg IoU(Norm): {i_s_n:.4f} | Seg IoU(Anom): {i_s_a:.4f} | MaxP: {max_p:.4f} | Px@0.1: {int(px01)}")

        plot_learning_curves(history)
        visualize_final_predictions(model, dl_te_d, dl_te_s, device, prefix="final_test")
        
        # === 獨立重繪包含真實瑕疵的影像 ===
        # 由於原始抽取可能抽到良品，此函式確保抽出含瑕疵的樣本繪圖
        # 如果模型已保存權重，您也可以在訓練腳本外獨立呼叫此函式
        redraw_defect_visualizations(model, test_det, full_seg, device, prefix="redraw")
        
        td_a, ts_i_n, ts_i_a, _, _, _ = evaluate(model, dl_te_d, dl_te_s, device)
        print("\n" + "="*50 + f"\n最終測試報告: Det Acc: {td_a:.6f}, Seg IoU(Norm): {ts_i_n:.4f}, Seg IoU(Anom): {ts_i_a:.4f}\n" + "="*50)
            
    except Exception as e: 
        import traceback
        traceback.print_exc() # 提供更詳細的報錯資訊