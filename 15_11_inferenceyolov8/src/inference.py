from ultralytics import YOLO
import cv2

class YOLOv8inference:
    '''
        A class for performing inference using the YOLOv8 model on images, videos, webcams stream.

    '''

    def __init__(self, model_path: str):
        '''
            Initialize the YOLOv8Inference class.
        Args:
            model_path (str): Path to the YOLOv8 model file.
        '''
        self.model_path = model_path
        self.model = YOLO(model_path)


    def infer_image(self, image_path: str, output_path: str):
        """
        Performs inference on a single image and saves the results.

        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the resulting image with detections.
        """
        results = self.model.predict(source=image_path, save=False)

        annotated_image = results[0].plot()
        cv2.imwrite(output_path, annotated_image)

        print(f"Image inference completed. Results saved to {output_path}")


    def infer_video(self, video_path: str, output_path: str):
        """
            Performs inference on a video file and saves annotated video

            Args:
                video_path (str): Path to the input video file.
                output_path (str): Path to save the resulting video with detections.
        """ 

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(source=frame, save=False)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        print(f"Video inference completed, Results saved to {output_path}")

    
    def infer_webcam(self, webcam_index: int = 0):
        '''
            Performs inference on a webcam stream in realtime

            Args:
                webcam_index: index of the webcam (Default = 0)
        '''

        cap = cv2.VideoCapture(webcam_index)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            results = self.model.predict(source=frame, save=False)
            annotated_frame = results[0].plot()

            cv2.imshow('Webcam inference', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting webcam inference...")
                break

        cap.release()
        cv2.destroyAllWindows()

