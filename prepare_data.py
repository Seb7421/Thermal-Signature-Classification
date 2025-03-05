import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import random
import json
import shutil
from tqdm import tqdm

class BalancedDatasetCreator:
    def __init__(self, base_dir=".", target_size=500, train_ratio=0.8):
        """
        Create a balanced dataset for human vs animal classification.
        
        Args:
            base_dir: Base directory where all data is stored
            target_size: Target number of images for the final dataset
            train_ratio: Ratio of training images (0.8 = 80% train, 20% test)
        """
        self.base_dir = Path(base_dir)
        self.target_size = target_size
        self.train_ratio = train_ratio
        
        # Input paths
        self.human_anno_dir = self.base_dir / "human_annotations"
        self.unknown_anno_dir = self.base_dir / "unkown_animal_annotations"
        self.elephant_anno_dir = self.base_dir / "elephant_annotations"
        
        self.human_img_dir = self.base_dir / "human_images"
        self.unknown_img_dir = self.base_dir / "unknown_animal_images"
        self.elephant_img_dir = self.base_dir / "elephant_images"
        
        # Output paths
        self.output_dir = self.base_dir / "fair_dataset"
        (self.output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "images").mkdir(parents=True, exist_ok=True)
        
        # Column names for the CSV files
        self.columns = [
            "frame_number", "object_id", "x", "y", "w", "h", 
            "class", "species", "occlusion", "noise"
        ]
        
        # For tracking matched annotations
        self.human_images = []
        self.animal_images = []
        
        # For statistics
        self.stats = {
            "total_found": 0,
            "human_found": 0,
            "animal_found": 0,
            "included_human": 0,
            "included_animal": 0,
            "train_images": 0,
            "test_images": 0,
            "train_annotations": 0,
            "test_annotations": 0,
            "human_annotations": 0,
            "animal_annotations": 0,
            "species_distribution": {}
        }
    
    def find_all_images(self):
        """Find all images with annotations and categorize them."""
        print("Finding all annotated images...")
        
        # Process human annotations
        for anno_file in self.human_anno_dir.glob("*.csv"):
            sequence_id = anno_file.stem
            sequence_dir = self.human_img_dir / sequence_id
            
            if not sequence_dir.exists():
                print(f"Warning: Directory not found for {sequence_id}")
                continue
                
            df = pd.read_csv(anno_file, header=None, names=self.columns)
            
            # Group by frame number to get one entry per image
            for frame_id, frame_df in df.groupby("frame_number"):
                # Look for the image file
                pattern = f"{sequence_id}*{int(frame_id):010d}*.jpg"
                matching_images = list(sequence_dir.glob(pattern))
                
                if matching_images:
                    image_path = matching_images[0]
                    
                    # Check that the image is valid
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        # Ensure all annotations have class=1 (human)
                        frame_df_fixed = frame_df.copy()
                        frame_df_fixed["class"] = 1
                        
                        # Store this image as a human image
                        self.human_images.append({
                            "image_path": image_path,
                            "annotations": frame_df_fixed.to_dict("records"),
                            "source": "human"
                        })
        
        # Process unknown animal annotations
        for anno_file in self.unknown_anno_dir.glob("*.csv"):
            sequence_id = anno_file.stem
            sequence_dir = self.unknown_img_dir / sequence_id
            
            if not sequence_dir.exists():
                print(f"Warning: Directory not found for {sequence_id}")
                continue
                
            df = pd.read_csv(anno_file, header=None, names=self.columns)
            
            # Group by frame number to get one entry per image
            for frame_id, frame_df in df.groupby("frame_number"):
                # Look for the image file
                pattern = f"{sequence_id}*{int(frame_id):010d}*.jpg"
                matching_images = list(sequence_dir.glob(pattern))
                
                if matching_images:
                    image_path = matching_images[0]
                    
                    # Check that the image is valid
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        # Ensure all annotations have class=0 (animal)
                        frame_df_fixed = frame_df.copy()
                        frame_df_fixed["class"] = 0
                        
                        # Store this image as an animal image
                        self.animal_images.append({
                            "image_path": image_path,
                            "annotations": frame_df_fixed.to_dict("records"),
                            "source": "unknown_animal"
                        })
        
        # Process elephant annotations
        for anno_file in self.elephant_anno_dir.glob("*.csv"):
            sequence_id = anno_file.stem
            sequence_dir = self.elephant_img_dir / sequence_id
            
            if not sequence_dir.exists():
                print(f"Warning: Directory not found for {sequence_id}")
                continue
                
            df = pd.read_csv(anno_file, header=None, names=self.columns)
            
            # Group by frame number to get one entry per image
            for frame_id, frame_df in df.groupby("frame_number"):
                # Look for the image file
                pattern = f"{sequence_id}*{int(frame_id):010d}*.jpg"
                matching_images = list(sequence_dir.glob(pattern))
                
                if matching_images:
                    image_path = matching_images[0]
                    
                    # Check that the image is valid
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        # Ensure all annotations have class=0 (animal)
                        frame_df_fixed = frame_df.copy()
                        frame_df_fixed["class"] = 0
                        
                        # Store this image as an animal image
                        self.animal_images.append({
                            "image_path": image_path,
                            "annotations": frame_df_fixed.to_dict("records"),
                            "source": "elephant"
                        })
        
        # Update statistics
        self.stats["human_found"] = len(self.human_images)
        self.stats["animal_found"] = len(self.animal_images)
        self.stats["total_found"] = self.stats["human_found"] + self.stats["animal_found"]
        
        print(f"Found {self.stats['total_found']} annotated images:")
        print(f"  - Human: {self.stats['human_found']} images")
        print(f"  - Animal: {self.stats['animal_found']} images")
    
    def create_dataset(self):
        """Create a balanced dataset with the specified target size."""
        print(f"Creating balanced dataset with {self.target_size} images...")
        
        # Determine how many images to include from each category
        half_target = self.target_size // 2
        num_humans = min(self.stats["human_found"], half_target)
        num_animals = min(self.stats["animal_found"], self.target_size - num_humans)
        
        # If we don't have enough of one category, use more of the other
        if num_humans < half_target:
            num_animals = min(self.stats["animal_found"], self.target_size - num_humans)
        if num_animals < (self.target_size - num_humans):
            num_humans = min(self.stats["human_found"], self.target_size - num_animals)
        
        total_selected = num_humans + num_animals
        
        # Select random images from each category
        selected_humans = random.sample(self.human_images, num_humans)
        selected_animals = random.sample(self.animal_images, num_animals)
        
        # Update statistics
        self.stats["included_human"] = num_humans
        self.stats["included_animal"] = num_animals
        
        # Combine and shuffle
        all_images = selected_humans + selected_animals
        random.shuffle(all_images)
        
        # Split into train and test
        split_index = int(len(all_images) * self.train_ratio)
        train_images = all_images[:split_index]
        test_images = all_images[split_index:]
        
        # Update split statistics
        self.stats["train_images"] = len(train_images)
        self.stats["test_images"] = len(test_images)
        
        print(f"Selected {total_selected} images ({num_humans} human, {num_animals} animal)")
        print(f"Split into {len(train_images)} training and {len(test_images)} test images")
        
        # Process train images
        train_annotations = []
        for i, img_data in enumerate(tqdm(train_images, desc="Processing train images")):
            # Create output image name and path
            img_filename = f"image_{i:06d}.jpg"
            img_output_path = self.output_dir / "train" / "images" / img_filename
            
            # Copy the image
            img = cv2.imread(str(img_data["image_path"]))
            cv2.imwrite(str(img_output_path), img)
            
            # Process annotations
            for anno in img_data["annotations"]:
                # Create a copy with updated frame number
                anno_copy = anno.copy()
                anno_copy["frame_number"] = i
                train_annotations.append(anno_copy)
                
                # Update statistics
                if anno_copy["class"] == 1:
                    self.stats["human_annotations"] += 1
                else:
                    self.stats["animal_annotations"] += 1
                
                # Update species statistics
                species = anno_copy["species"]
                if species not in self.stats["species_distribution"]:
                    self.stats["species_distribution"][species] = 0
                self.stats["species_distribution"][species] += 1
                
        # Track total annotations
        self.stats["train_annotations"] = len(train_annotations)
        
        # Save train annotations
        train_df = pd.DataFrame(train_annotations)
        train_df.to_csv(self.output_dir / "train" / "annotations.csv", index=False)
        
        # Process test images
        test_annotations = []
        for i, img_data in enumerate(tqdm(test_images, desc="Processing test images")):
            # Create output image name and path
            img_filename = f"image_{i:06d}.jpg"
            img_output_path = self.output_dir / "test" / "images" / img_filename
            
            # Copy the image
            img = cv2.imread(str(img_data["image_path"]))
            cv2.imwrite(str(img_output_path), img)
            
            # Process annotations
            for anno in img_data["annotations"]:
                # Create a copy with updated frame number
                anno_copy = anno.copy()
                anno_copy["frame_number"] = i
                test_annotations.append(anno_copy)
                
                # Update statistics
                if anno_copy["class"] == 1:
                    self.stats["human_annotations"] += 1
                else:
                    self.stats["animal_annotations"] += 1
                
                # Update species statistics
                species = anno_copy["species"]
                if species not in self.stats["species_distribution"]:
                    self.stats["species_distribution"][species] = 0
                self.stats["species_distribution"][species] += 1
        
        # Track total annotations
        self.stats["test_annotations"] = len(test_annotations)
        
        # Save test annotations
        test_df = pd.DataFrame(test_annotations)
        test_df.to_csv(self.output_dir / "test" / "annotations.csv", index=False)
        
        # Calculate image sources
        source_counts = {
            "human": sum(1 for img in all_images if img["source"] == "human"),
            "unknown_animal": sum(1 for img in all_images if img["source"] == "unknown_animal"),
            "elephant": sum(1 for img in all_images if img["source"] == "elephant")
        }
        
        # Calculate average annotations per image
        avg_annotations = (self.stats["train_annotations"] + self.stats["test_annotations"]) / total_selected
        
        # Save metadata with clear differentiation between image and annotation counts
        metadata = {
            "target_size": self.target_size,
            "actual_dataset_size": total_selected,
            "train_images": len(train_images),
            "test_images": len(test_images),
            "human_images": num_humans,
            "animal_images": num_animals,
            
            "total_annotations": self.stats["train_annotations"] + self.stats["test_annotations"],
            "train_annotations": self.stats["train_annotations"],
            "test_annotations": self.stats["test_annotations"],
            "human_annotations": self.stats["human_annotations"],
            "animal_annotations": self.stats["animal_annotations"],
            "avg_annotations_per_image": round(avg_annotations, 2),
            
            "image_sources": source_counts,
            "species_distribution": self.stats["species_distribution"]
        }
        
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("Dataset creation completed successfully!")
        print(f"Total images: {total_selected}")
        print(f"Total annotations: {self.stats['train_annotations'] + self.stats['test_annotations']}")
        print(f"Average annotations per image: {avg_annotations:.2f}")
    
    def create_visualizations(self, num_samples=10):
        """Create visualizations of dataset samples."""
        print("Creating visualization samples...")
        
        vis_dir = self.output_dir / "visualization"
        vis_dir.mkdir(exist_ok=True)
        
        # Read train annotations
        train_df = pd.read_csv(self.output_dir / "train" / "annotations.csv")
        
        # Get unique frame numbers
        unique_frames = train_df["frame_number"].unique()
        
        # Select random frames
        if len(unique_frames) > num_samples:
            sample_frames = random.sample(list(unique_frames), num_samples)
        else:
            sample_frames = unique_frames
        
        # Class colors
        colors = {
            0: (0, 0, 255),  # Red for animals
            1: (0, 255, 0)   # Green for humans
        }
        
        # Species names
        species_names = {
            -1: "unknown", 0: "human", 1: "elephant", 2: "lion", 
            3: "giraffe", 4: "dog", 5: "crocodile", 6: "hippo", 
            7: "zebra", 8: "rhino"
        }
        
        # Create visualizations
        for frame_num in sample_frames:
            # Get annotations for this frame
            frame_annos = train_df[train_df["frame_number"] == frame_num]
            
            # Load image
            img_path = self.output_dir / "train" / "images" / f"image_{int(frame_num):06d}.jpg"
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue
            
            # Draw annotations
            for _, anno in frame_annos.iterrows():
                x, y, w, h = int(anno["x"]), int(anno["y"]), int(anno["w"]), int(anno["h"])
                class_id = int(anno["class"])
                species_id = int(anno["species"])
                
                # Draw bounding box
                color = colors[class_id]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                class_name = "Human" if class_id == 1 else "Animal"
                species_name = species_names.get(species_id, f"Species {species_id}")
                label = f"{class_name} ({species_name})"
                
                # Add occlusion and noise info if present
                if anno["occlusion"] == 1:
                    label += ", occluded"
                if anno["noise"] == 1:
                    label += ", noisy"
                
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save the visualization
            vis_path = vis_dir / f"sample_{int(frame_num):06d}.jpg"
            cv2.imwrite(str(vis_path), img)
        
        print(f"Created {len(sample_frames)} visualization samples")
    
    def run(self):
        """Run the full dataset creation process."""
        self.find_all_images()
        self.create_dataset()
        self.create_visualizations()
        print("All done!")

# Create the dataset
if __name__ == "__main__":
    creator = BalancedDatasetCreator(target_size=500)
    creator.run()