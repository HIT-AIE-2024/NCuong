import os
from pathlib import Path
from src.inference import YOLOv8inference

if __name__ == '__main__':
    image_path = "15_11_inferenceyolov8/asset/image.jpg"
    video_path = "15_11_inferenceyolov8/asset/video.mp4"
    output_dir = "15_11_inferenceyolov8/output/" 
    os.makedirs(output_dir, exist_ok=True)

    inference =  YOLOv8inference('yolov8n.pt')


    # Example images
    image_name = Path(image_path).name
    output_image_path = os.path.join(output_dir, image_name)
    inference.infer_image(image_path, output_image_path)

    # Example video
    video_name = Path(video_path).name
    output_video_path = os.path.join(output_dir, video_name)
    inference.infer_video(video_path, output_video_path)

    # Example Webcam
    webcam_index = 0
    inference.infer_webcam(0)

