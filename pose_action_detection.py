import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
import time
import sys
import os
import argparse

class PoseActionDetector:
    def __init__(self):
        try:
            print("Initializing YOLOv8 model...")
            self.yolo_model = YOLO('yolov8n.pt')
            print("YOLOv8 model initialized successfully")
            
            print("Initializing DeepSORT tracker...")
            self.tracker = DeepSort(max_age=30)
            print("DeepSORT tracker initialized successfully")
            
            print("Initializing MediaPipe pose...")
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            print("MediaPipe pose initialized successfully")
            
            self.prev_pose_landmarks = None
            self.action_counter = 0
            self.action_threshold = 10
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def detect_pose(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
                return results.pose_landmarks
            return None
        except Exception as e:
            print(f"Error in detect_pose: {str(e)}")
            return None

    def detect_action(self, current_landmarks):
        try:
            if self.prev_pose_landmarks is None:
                self.prev_pose_landmarks = current_landmarks
                return "No action detected"

            current_shoulder = current_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            current_wrist = current_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            prev_shoulder = self.prev_pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            prev_wrist = self.prev_pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            if current_wrist.y < current_shoulder.y and prev_wrist.y >= prev_shoulder.y:
                self.action_counter += 1
                if self.action_counter >= self.action_threshold:
                    self.action_counter = 0
                    return "Raise hand detected"
            
            self.prev_pose_landmarks = current_landmarks
            return "No action detected"
        except Exception as e:
            print(f"Error in detect_action: {str(e)}")
            return "Error in action detection"

    def process_frame(self, frame):
        try:
            results = self.yolo_model(frame)[0]
            
            detections = []
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                if score > 0.5:
                    detections.append(([x1, y1, x2, y2], score, 'person'))
            
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            pose_landmarks = self.detect_pose(frame)
            action = "No action detected"
            if pose_landmarks:
                action = self.detect_action(pose_landmarks)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                
                cv2.rectangle(frame, 
                            (int(ltrb[0]), int(ltrb[1])), 
                            (int(ltrb[2]), int(ltrb[3])), 
                            (0, 255, 0), 2)
                
                cv2.putText(frame, f"ID: {track_id}", 
                           (int(ltrb[0]), int(ltrb[1])-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.putText(frame, action, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return frame
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            return frame

def process_image(image_path, output_dir):
    try:
        if not os.path.isfile(image_path):
            print(f"Error: Image file '{image_path}' does not exist")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image '{image_path}'")
            return False
        
        detector = PoseActionDetector()
        processed_frame = detector.process_frame(frame)
        
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        cv2.imwrite(output_path, processed_frame)
        print(f"Saved processed image to: {output_path}")
        
        return True
    except Exception as e:
        print(f"Error processing image '{image_path}': {str(e)}")
        return False

def process_directory(input_dir, output_dir):
    try:
        if not os.path.isdir(input_dir):
            print(f"Error: Input directory '{input_dir}' does not exist")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in os.listdir(input_dir) 
                      if os.path.isfile(os.path.join(input_dir, f)) and 
                      os.path.splitext(f)[1].lower() in image_extensions]
        
        if not image_files:
            print(f"No image files found in '{input_dir}'")
            return False
        
        success_count = 0
        for image_file in image_files:
            image_path = os.path.join(input_dir, image_file)
            if process_image(image_path, output_dir):
                success_count += 1
        
        print(f"Processed {success_count} out of {len(image_files)} images")
        return success_count > 0
    except Exception as e:
        print(f"Error processing directory '{input_dir}': {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Process images for pose action detection')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('--output', '-o', default='output', help='Output directory (default: output)')
    args = parser.parse_args()
    
    try:
        if os.path.isfile(args.input):
            process_image(args.input, args.output)
        elif os.path.isdir(args.input):
            process_directory(args.input, args.output)
        else:
            print(f"Error: Input '{args.input}' is not a valid file or directory")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()