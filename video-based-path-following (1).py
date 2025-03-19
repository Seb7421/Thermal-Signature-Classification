import cv2
import numpy as np
import time
import pickle
import os
import ipywidgets.widgets as widgets
from IPython.display import display
import motors
from zed import Camera
import traitlets
from sklearn.ensemble import RandomForestClassifier
from collections import deque

# ===============================
# PART 1: VIDEO PROCESSING TOOLS
# ===============================

def extract_frames(video_path, output_dir="temp_frames", sample_rate=10):
    """
    Extract frames from a video file at a specified sample rate.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        sample_rate: Extract every Nth frame
    
    Returns:
        List of paths to saved frames
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    frame_count = 0
    saved_frames = []
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        # Break the loop if we reached the end of the video
        if not ret:
            break
        
        # Extract frames at the specified sample rate
        if frame_count % sample_rate == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    
    print(f"Extracted {len(saved_frames)} frames from {video_path}")
    return saved_frames

def process_frame(frame, debug=False):
    """
    Process a frame to detect the yellow rope and extract features.
    
    Args:
        frame: Input frame (BGR format)
        debug: Whether to return debug images
    
    Returns:
        Dictionary of extracted features and processed images if debug=True
    """
    # Get image dimensions
    height, width = frame.shape[:2]
    
    # Define ROI (region of interest) - bottom section of the image
    roi_height = int(height * 0.3)  # Use bottom 30% of the image
    x_start = int(width * 0.3)  # Crop 30% from the left
    x_end = int(width * 0.7)    # Crop 30% from the right
    
    roi = frame[height - roi_height:, x_start:x_end]
    
    # Convert to HSV color space for better color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define yellow color range
    lower_yellow = np.array([10, 80, 80])
    upper_yellow = np.array([40, 255, 255])
    
    # Create mask for yellow regions
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Clean the mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = {}
    features['has_rope'] = len(contours) > 0
    
    # If contours found, extract features
    if features['has_rope']:
        # Find the largest contour (likely the rope)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get moments to find center of rope
        M = cv2.moments(largest_contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Calculate position relative to center of ROI
            roi_center_x = roi.shape[1] // 2
            
            # Calculate error (negative means rope is to the right of center)
            error = roi_center_x - cx
            
            # Calculate rope orientation using PCA
            rope_points = np.array([p[0] for p in largest_contour])
            if len(rope_points) >= 2:  # Need at least 2 points for PCA
                mean = np.mean(rope_points, axis=0)
                # Calculate covariance matrix
                cov = np.cov(rope_points.T)
                # Get eigenvectors and eigenvalues
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                # Get index of largest eigenvalue
                index = np.argmax(eigenvalues)
                # Get the corresponding eigenvector
                direction = eigenvectors[:, index]
                # Calculate angle in degrees
                angle = np.degrees(np.arctan2(direction[1], direction[0]))
            else:
                angle = 0
            
            # Store extracted features
            features['center_x'] = cx
            features['center_y'] = cy
            features['error'] = error
            features['angle'] = angle
            features['area'] = cv2.contourArea(largest_contour)
            
            # Draw detected center on the ROI for debugging
            if debug:
                debug_roi = roi.copy()
                cv2.circle(debug_roi, (cx, cy), 5, (0, 0, 255), -1)
                cv2.drawContours(debug_roi, [largest_contour], 0, (0, 255, 0), 2)
                cv2.line(debug_roi, (roi_center_x, 0), (roi_center_x, roi.shape[0]), (255, 0, 0), 2)
                features['debug_roi'] = debug_roi
                features['debug_mask'] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            features['has_rope'] = False
    
    features['roi_width'] = roi.shape[1]
    
    return features

# ========================
# PART 2: TRAINING SYSTEM
# ========================

class PathFollowingTrainer:
    def __init__(self):
        """Initialize the path following trainer."""
        self.dataset = []
        self.model = None
        self.command_mapping = {
            'forward': 0,
            'left': 1,
            'right': 2
        }
        self.inverse_command_mapping = {v: k for k, v in self.command_mapping.items()}
    
    def extract_features_from_video(self, video_path, temp_dir="temp_frames", sample_rate=10):
        """
        Process a video and extract features from each frame.
        
        Args:
            video_path: Path to the video file
            temp_dir: Directory to temporarily store extracted frames
            sample_rate: Extract every Nth frame
        
        Returns:
            List of features extracted from each frame
        """
        frame_paths = extract_frames(video_path, temp_dir, sample_rate)
        features_list = []
        
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Could not read frame {frame_path}")
                continue
                
            features = process_frame(frame)
            features['frame_path'] = frame_path  # Store reference to the frame
            features_list.append(features)
        
        return features_list
    
    def label_dataset(self, features_list, auto_label=True, label_straight_sections=True):
        """
        Label the dataset with appropriate motor commands.
        
        Args:
            features_list: List of features extracted from frames
            auto_label: Whether to automatically label based on features
            label_straight_sections: Whether to label straight sections
        
        Returns:
            List of (features, label) pairs
        """
        labeled_data = []
        
        for i, features in enumerate(features_list):
            if not features['has_rope']:
                continue
            
            # Default command
            command = 'forward'
            
            if auto_label:
                error = features.get('error', 0)
                angle = features.get('angle', 0)
                
                # Enhanced rule-based labeling that considers both position and angle
                if abs(error) < 50:  # Center of path is near the center of ROI
                    command = 'forward'
                elif error > 0:  # Rope is to the left of center
                    command = 'left'
                    # If angle indicates path is curving right despite being left of center, 
                    # prioritize the angle for sharp turns
                    if -90 < angle < -45:
                        command = 'right'
                else:  # Rope is to the right of center
                    command = 'right'
                    # Similar logic for the opposite direction
                    if 45 < angle < 90:
                        command = 'left'
            
            # Convert features to numerical format suitable for ML
            feature_vector = self._extract_feature_vector(features)
            labeled_data.append((feature_vector, self.command_mapping[command]))
        
        return labeled_data
    
    def _extract_feature_vector(self, features):
        """Extract a numerical feature vector from the features dictionary."""
        if not features['has_rope']:
            return np.array([0, 0, 0, 0])
        
        # Normalize error by ROI width
        normalized_error = features['error'] / features['roi_width']
        
        return np.array([
            normalized_error,
            features.get('angle', 0) / 180.0,  # Normalize angle
            features.get('area', 0) / (features['roi_width'] ** 2),  # Normalize area
            1.0  # Indicator that rope is detected
        ])
    
    def train_model(self, labeled_data):
        """
        Train a machine learning model on the labeled data.
        
        Args:
            labeled_data: List of (feature_vector, label) pairs
        """
        if not labeled_data:
            print("Error: No labeled data available for training")
            return False
        
        # Split into features and labels
        X = np.array([features for features, _ in labeled_data])
        y = np.array([label for _, label in labeled_data])
        
        # Print some stats
        print(f"Command distribution:")
        for cmd_id, count in zip(*np.unique(y, return_counts=True)):
            cmd_name = self.inverse_command_mapping[cmd_id]
            print(f"  {cmd_name}: {count} samples ({count/len(y)*100:.1f}%)")
        
        # Train a Random Forest classifier with more trees for better accuracy
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        print(f"Model trained on {len(labeled_data)} samples")
        return True
    
    def save_model(self, model_path):
        """Save the trained model to a file."""
        if self.model is None:
            print("Error: No trained model to save")
            return False
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to {model_path}")
        return True
    
    def load_model(self, model_path):
        """Load a trained model from a file."""
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} does not exist")
            return False
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Model loaded from {model_path}")
        return True

# ===========================
# PART 3: EXECUTION SYSTEM
# ===========================

class VideoBasedPathFollower:
    def __init__(self, model_path, robot=None):
        """
        Initialize the video-based path follower.
        
        Args:
            model_path: Path to the trained model file
            robot: Robot motor controller instance
        """
        self.model = None
        self.robot = robot
        self.command_mapping = {
            0: 'forward',
            1: 'left',
            2: 'right'
        }
        self.base_speed = 0.5
        self.turn_speed = 0.2
        self.last_commands = deque(maxlen=5)  # Store recent commands for smoothing
        
        # Load the model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def process_and_control(self, frame):
        """
        Process a frame and control the robot accordingly.
        
        Args:
            frame: Input camera frame
        
        Returns:
            Processed frame with visualization and command issued
        """
        # Extract features from the frame
        features = process_frame(frame, debug=True)
        
        # Create a copy for visualization
        vis_frame = frame.copy()
        
        # Default command and visualization text
        command = None
        text = "No rope detected"
        color = (0, 0, 255)  # Red
        
        if features['has_rope']:
            # Extract feature vector for prediction
            feature_vector = np.array([
                features['error'] / features['roi_width'],
                features.get('angle', 0) / 180.0,
                features.get('area', 0) / (features['roi_width'] ** 2),
                1.0
            ]).reshape(1, -1)
            
            # Predict command
            prediction = self.model.predict(feature_vector)[0]
            command = self.command_mapping[prediction]
            
            # Implement command smoothing
            self.last_commands.append(command)
            
            # Use majority vote for smoothing
            commands_count = {cmd: self.last_commands.count(cmd) for cmd in set(self.last_commands)}
            smoothed_command = max(commands_count, key=commands_count.get)
            
            # Execute command if robot is available
            if self.robot:
                self._execute_command(smoothed_command, features['error'])
            
            text = f"Command: {smoothed_command.upper()} | Error: {features['error']}"
            color = (0, 255, 0)  # Green
            
            # Add visualization
            height, width = frame.shape[:2]
            roi_height = int(height * 0.3)
            x_start = int(width * 0.3)
            x_end = int(width * 0.7)
            
            # Draw ROI rectangle
            cv2.rectangle(vis_frame, (x_start, height - roi_height), (x_end, height), (255, 0, 0), 2)
            
            # Draw center line
            cv2.line(vis_frame, (width // 2, height - roi_height), (width // 2, height), (0, 0, 255), 1)
            
            # Draw detected rope center
            if 'center_x' in features and 'center_y' in features:
                center_x = features['center_x'] + x_start
                center_y = features['center_y'] + (height - roi_height)
                cv2.circle(vis_frame, (center_x, center_y), 5, (0, 255, 255), -1)
        
        # Add text with command information
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_frame, command
    
    def _execute_command(self, command, error=0):
        """
        Execute a command on the robot.
        
        Args:
            command: Command to execute ('forward', 'left', 'right')
            error: Error value for proportional control
        """
        if self.robot is None:
            return
        
        # Dynamic speed adjustment based on error magnitude
        # Higher error = sharper turn needed
        error_magnitude = abs(error)
        
        if command == 'forward':
            self.robot.forward(speed=self.base_speed)
        elif command == 'left':
            # Progressive turn speed based on error magnitude
            turn_speed = min(0.4, self.turn_speed + (error_magnitude / 500.0) * 0.2)
            self.robot.left(speed=turn_speed)
        elif command == 'right':
            # Progressive turn speed based on error magnitude
            turn_speed = min(0.4, self.turn_speed + (error_magnitude / 500.0) * 0.2)
            self.robot.right(speed=turn_speed)

# ===========================
# PART 4: WRAPPER FUNCTIONS
# ===========================

def train_from_video(video_path, model_save_path, sample_rate=10):
    """
    Train a path following model from a video file.
    
    Args:
        video_path: Path to the training video
        model_save_path: Path to save the trained model
        sample_rate: Take every Nth frame from the video
    """
    # Initialize trainer
    trainer = PathFollowingTrainer()
    
    # Extract features from video
    print(f"Processing video: {video_path}")
    features_list = trainer.extract_features_from_video(video_path, sample_rate=sample_rate)
    
    # Check if we extracted valid features
    if not features_list:
        print("Error: No valid frames could be extracted from the video")
        return False
        
    # Count frames with visible rope
    rope_visible_frames = sum(1 for features in features_list if features['has_rope'])
    print(f"Found {rope_visible_frames} frames with visible rope out of {len(features_list)} total frames")
    
    if rope_visible_frames < 10:
        print("Warning: Very few frames with visible rope. Check video quality and color thresholds.")
        if rope_visible_frames == 0:
            print("Cannot proceed with training as no rope was detected.")
            return False
    
    # Label the dataset
    print("Labeling dataset...")
    labeled_data = trainer.label_dataset(features_list, auto_label=True)
    
    # Train the model
    print("Training model...")
    if trainer.train_model(labeled_data):
        # Save the trained model
        trainer.save_model(model_save_path)
        print(f"Training complete! Model saved to {model_save_path}")
        return True
    else:
        print("Training failed.")
        return False

def run_path_follower(model_path, use_camera=True, video_path=None):
    """
    Run the path follower using either live camera feed or a pre-recorded video.
    
    Args:
        model_path: Path to the trained model
        use_camera: Whether to use live camera feed (True) or pre-recorded video (False)
        video_path: Path to the pre-recorded video (required if use_camera=False)
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        return
    
    # Initialize the robot
    robot = motors.MotorsYukon(mecanum=False) if use_camera else None
    
    # Initialize path follower
    path_follower = VideoBasedPathFollower(model_path, robot)
    
    # Set up display widgets for visualization
    display_original = widgets.Image(format='jpeg', width='320px')
    display_processed = widgets.Image(format='jpeg', width='320px')
    layout = widgets.Layout(width='100%')
    sidebyside = widgets.HBox([display_original, display_processed], layout=layout)
    display(sidebyside)
    
    def bgr8_to_jpeg(image):
        """Convert a BGR image to JPEG format for display."""
        return bytes(cv2.imencode('.jpg', image)[1])
    
    if use_camera:
        # Use live camera feed
        camera = Camera.instance()
        camera.start()
        
        def process_camera_frame(change):
            frame = change['new']
            if frame is None:
                return
            
            # Process the frame and control the robot
            processed_frame, _ = path_follower.process_and_control(frame)
            
            # Update displays
            display_original.value = bgr8_to_jpeg(cv2.resize(frame, (320, 240)))
            display_processed.value = bgr8_to_jpeg(cv2.resize(processed_frame, (320, 240)))
        
        # Start observing camera frames
        camera.observe(process_camera_frame, names='color_value')
        
        print("Path following started. Press Ctrl+C in the terminal to stop.")
    else:
        # Use pre-recorded video
        if video_path is None:
            print("Error: Video path required when not using camera")
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video reached")
                break
            
            # Process the frame
            processed_frame, _ = path_follower.process_and_control(frame)
            
            # Update displays
            display_original.value = bgr8_to_jpeg(cv2.resize(frame, (320, 240)))
            display_processed.value = bgr8_to_jpeg(cv2.resize(processed_frame, (320, 240)))
            
            # Add a small delay to simulate real-time processing
            time.sleep(0.1)
        
        cap.release()



# STEP 1: Define the path to your video
VIDEO_PATH = 'color_video.avi'  # Your video is in the same directory
MODEL_PATH = 'rope_follower_model.pkl'  # Where to save the trained model

# STEP 2: Train the model using your video
# You can adjust sample_rate to control how many frames are used (lower = more frames)
print("Starting training process...")
train_from_video(VIDEO_PATH, MODEL_PATH, sample_rate=5)

# STEP 3: Test the model on the video (visualization only, no robot control)
print("\nTesting model on video...")
run_path_follower(MODEL_PATH, use_camera=False, video_path=VIDEO_PATH)

# STEP 4: Uncomment the line below when you're ready to run on the real robot
# print("\nRunning on the robot...")
# run_path_follower(MODEL_PATH, use_camera=True)
