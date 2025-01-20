import torchvision
from torchvision import transforms
from models.dino2 import DINO2SEG
from utils.mscoco import COCOSegmentation
from utils.segmentationMetric import *
from utils.vis import decode_segmap
import numpy as np
import torch
import os
from PIL import Image, ImageOps
from simple_colors import green
from tqdm import tqdm


@torch.inference_mode()
def save_preds(split, modelname, model_path, model_big=False, read_from_saved=False, dataset="safety"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dirpath = f"/robodata/smodak/corl_rebuttal/dino_traindata/{dataset}/{split}/images"
    gt_preds_dir = f"/robodata/smodak/corl_rebuttal/dino_traindata/{dataset}/{split}/gt_preds"
    pred_root_dir = f"/robodata/smodak/corl_rebuttal/seg-dinov2/mypreds/{dataset}/{split}/{modelname}"
    os.makedirs(pred_root_dir, exist_ok=True)
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    model = DINO2SEG(3, big=model_big).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()

    mious = []
    iou_pos = []
    iou_neg = []
    all_images = os.listdir(images_dirpath)
    for image_name in tqdm(all_images, desc=f"Processing {split}", total=len(all_images)):
        noext_img_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(images_dirpath, image_name)
        pred_savepath = os.path.join(pred_root_dir, f"{noext_img_name}.bin")
        img = Image.open(image_path).convert('RGB')
        if not read_from_saved:
            padding = (0, 210, 0, 210)  # left, top, right, bottom
            img = ImageOps.expand(img, padding, fill=0)
            img = img.resize((540, 540), Image.BILINEAR)
            w, h = img.size
            short_size = 448  # crop_size
            if w > h:
                oh = short_size
                ow = int(1.0 * w * oh / h)
            else:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            img = img.resize((ow, oh), Image.BILINEAR)
            img = np.array(img)
            img = input_transform(img).unsqueeze(0).to(DEVICE)
            out = model(img)
            upsampled_logits = torch.nn.functional.interpolate(
                out,
                size=(960, 960),
                mode='bilinear',
                align_corners=False
            )
            pred_seg = upsampled_logits.argmax(dim=1)[0]
            reduced_predseg = pred_seg[210:750, :].cpu().numpy()
        else:
            reduced_predseg = np.fromfile(pred_savepath, dtype=np.uint8).reshape((540, 960))
        gt_seg = np.fromfile(os.path.join(gt_preds_dir, f"{noext_img_name}.bin"), dtype=np.uint8).reshape((540, 960))
        calc_iou = calculate_iou(gt_seg, reduced_predseg)
        mious.append(calc_iou['miou'])
        iou_pos.append(calc_iou[1])
        iou_neg.append(calc_iou[2])
        flat_nn_seg_np = reduced_predseg.reshape(-1).astype(np.uint8)
        flat_nn_seg_np.tofile(pred_savepath)
    mean_iou_pos = np.mean(iou_pos) * 100
    mean_miou = np.mean(mious) * 100
    mean_iou_neg = 2 * mean_miou - mean_iou_pos
    print(green(f"Pos IoU for {split} is {mean_iou_pos}", ['bold']))
    print(green(f"Neg IoU for {split} is {mean_iou_neg}", ['bold']))
    print(green(f"Mean IoU for {split} is {mean_miou}", ['bold']))
    return mious


@torch.inference_mode()
def calculate_iou(gt_mask, pred_mask, num_classes=3):
    # Convert the masks to torch tensors if they are not already
    gt_tensor = torch.tensor(gt_mask, dtype=torch.int64)
    pred_tensor = torch.tensor(pred_mask, dtype=torch.int64)
    # Initialize a dictionary to store IoU for each class
    iou_scores = {}
    # Calculate IoU for each class
    for cls in range(num_classes):
        # Create binary masks for the current class
        gt_cls = (gt_tensor == cls)
        pred_cls = (pred_tensor == cls)
        # Calculate intersection and union
        intersection = torch.logical_and(gt_cls, pred_cls).sum().item()
        union = torch.logical_or(gt_cls, pred_cls).sum().item()
        # Compute IoU and handle division by zero
        if union == 0:
            iou_scores[cls] = float('nan')
        else:
            iou_scores[cls] = intersection / union
    # Calculate mean IoU
    valid_ious = [iou for iou in iou_scores.values() if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else float('nan')
    # Add mean IoU to the dictionary
    iou_scores['miou'] = miou
    return iou_scores


if __name__ == "__main__":
    mious = []
    for split in ["train", "test", "eval"]:
        mious += save_preds(split=split,
                            modelname="small",
                            model_path="/robodata/smodak/corl_rebuttal/seg-dinov2/mymodels/parking/small/dinov2_mscoco.pth",
                            model_big=False,
                            read_from_saved=True,
                            dataset="parking")
    # print(green(f"Mean IoU for all splits is {np.mean(mious)}", ['bold', 'underlined']))
    
    mious = []
    for split in ["train", "test", "eval"]:
        mious += save_preds(split=split,
                            modelname="giga",
                            model_path="/robodata/smodak/corl_rebuttal/seg-dinov2/mymodels/parking/giga/dinov2_mscoco.pth",
                            model_big=True,
                            read_from_saved=True,
                            dataset="parking")
    # print(green(f"Mean IoU for all splits is {np.mean(mious)}", ['bold', 'underlined']))
