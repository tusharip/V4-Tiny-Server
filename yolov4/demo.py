import cv2
import torch

from yolov4.utils import LoadImage, non_max_suppression, scale_coords, plot_fire

# from utils import LoadImage, non_max_suppression, scale_coords, plot_fire
# from model import *


def detect(model, source, out, imgsz, conf_thres, iou_thres,  
           names, colors=[(255, 30, 0), (50, 0, 255)], device=torch.device('cpu')):
    img, img0 = LoadImage(source, img_size=imgsz)

    # Run inference
    img, im0 = LoadImage(source, img_size=imgsz)
    img = torch.from_numpy(img).to(device)
    img = img.float() 
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Process detections
    det = pred[0]  # detections 
    if det is not None and len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class

        # Write results
        for *xyxy, conf, cls in det:
            label = '%s %.2f' % (names[int(cls)], conf)
            # if cls == 0:
            plot_fire(xyxy, im0, clas=cls, label=label, color=colors[int(cls)], line_thickness=2)

    # Save results (image with detections)
    cv2.imwrite(out, im0)

