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
    def __init__(self, base_dir=".", target_size=500, train_ratio=0.7, val_ratio=0.15):
        """
        Create a balanced dataset for human vs animal classification.
        
        Args:
            base_dir: Base directory where all data is stored
            target_size: Target number of images for the final dataset
            train_ratio: Ratio of training images (0.7 = 70% train)
            val_ratio: Ratio of validation images (0.15 = 15% validation, remaining 15% for test)
        """
        self.base_dir = Path(base_dir)
        self.target_size = target_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Input paths
        self.human_anno_dir = self.base_dir / "human_annotations"
        self.unknown_anno_dir = self.base_dir / "unkown_animal_annotations"
        self.elephant_anno_dir = self.base_dir / "elephant_annotations"
        
        self.human_img_dir = self.base_dir / "human_images"
        self.unknown_img_dir = self.base_dir / "unknown_animal_images"
        self.elephant_img_dir = self.base_dir / "elephant_images"
        
        # Output paths - now includes validation directory
        self.output_dir = self.base_dir / "birdsai_data"
        (self.output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
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
            "val_images": 0,
            "test_images": 0,
            "train_annotations": 0,
            "val_annotations": 0,
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
        """Create a balanced dataset with the specified target size split into train, val and test."""
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
        
        # Split into train, validation and test sets
        train_split = int(len(all_images) * self.train_ratio)
        val_split = int(len(all_images) * (self.train_ratio + self.val_ratio))
        
        train_images = all_images[:train_split]
        val_images = all_images[train_split:val_split]
        test_images = all_images[val_split:]
        
        # Update split statistics
        self.stats["train_images"] = len(train_images)
        self.stats["val_images"] = len(val_images)
        self.stats["test_images"] = len(test_images)
        
        print(f"Selected {total_selected} images ({num_humans} human, {num_animals} animal)")
        print(f"Split into {len(train_images)} training, {len(val_images)} validation, and {len(test_images)} test images")
        
        # Verify class distribution in each split
        train_humans = sum(1 for img in train_images if img["annotations"][0]["class"] == 1)
        train_animals = len(train_images) - train_humans
        
        val_humans = sum(1 for img in val_images if img["annotations"][0]["class"] == 1)
        val_animals = len(val_images) - val_humans
        
        test_humans = sum(1 for img in test_images if img["annotations"][0]["class"] == 1)
        test_animals = len(test_images) - test_humans
        
        print(f"Training split: {train_humans} humans, {train_animals} animals")
        print(f"Validation split: {val_humans} humans, {val_animals} animals")
        print(f"Test split: {test_humans} humans, {test_animals} animals")
        
        # Process training images
        self._process_split(train_images, "train")
        
        # Process validation images
        self._process_split(val_images, "val")
        
        # Process test images
        self._process_split(test_images, "test")
        
        # Calculate image sources
        source_counts = {
            "human": sum(1 for img in all_images if img["source"] == "human"),
            "unknown_animal": sum(1 for img in all_images if img["source"] == "unknown_animal"),
            "elephant": sum(1 for img in all_images if img["source"] == "elephant")
        }
        
        # Calculate average annotations per image
        total_annotations = (self.stats["train_annotations"] + 
                             self.stats["val_annotations"] + 
                             self.stats["test_annotations"])
        avg_annotations = total_annotations / total_selected
        
        # Save metadata with clear differentiation between image and annotation counts
        metadata = {
            "target_size": self.target_size,
            "actual_dataset_size": total_selected,
            "train_images": len(train_images),
            "val_images": len(val_images),
            "test_images": len(test_images),
            "human_images": num_humans,
            "animal_images": num_animals,
            
            "total_annotations": total_annotations,
            "train_annotations": self.stats["train_annotations"],
            "val_annotations": self.stats["val_annotations"],
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
        print(f"Total annotations: {total_annotations}")
        print(f"Average annotations per image: {avg_annotations:.2f}")
    
    def _process_split(self, images, split_name):
        """Process a split of images and annotations."""
        annotations = []
        for i, img_data in enumerate(tqdm(images, desc=f"Processing {split_name} images")):
            # Create output image name and path
            img_filename = f"image_{i:06d}.jpg"
            img_output_path = self.output_dir / split_name / "images" / img_filename
            
            # Copy the image
            img = cv2.imread(str(img_data["image_path"]))
            cv2.imwrite(str(img_output_path), img)
            
            # Process annotations
            for anno in img_data["annotations"]:
                # Create a copy with updated frame number
                anno_copy = anno.copy()
                anno_copy["frame_number"] = i
                annotations.append(anno_copy)
                
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
        self.stats[f"{split_name}_annotations"] = len(annotations)
        
        # Save annotations
        annotations_df = pd.DataFrame(annotations)
        annotations_df.to_csv(self.output_dir / split_name / "annotations.csv", index=False)
        
        print(f"Processed {split_name} split: {len(images)} images, {len(annotations)} annotations")
    
    def create_visualizations(self, num_samples=10):
        """Create visualizations of dataset samples."""
        print("Creating visualization samples...")
        
        vis_dir = self.output_dir / "visualization"
        vis_dir.mkdir(exist_ok=True)
        
        # Choose samples from all splits
        sample_selections = []
        
        # Get samples from train set
        train_df = pd.read_csv(self.output_dir / "train" / "annotations.csv")
        train_frames = train_df["frame_number"].unique()
        train_samples = min(num_samples // 3, len(train_frames))
        if train_samples > 0:
            train_selected = random.sample(list(train_frames), train_samples)
            sample_selections.extend([("train", frame) for frame in train_selected])
        
        # Get samples from val set
        val_df = pd.read_csv(self.output_dir / "val" / "annotations.csv")
        val_frames = val_df["frame_number"].unique()
        val_samples = min(num_samples // 3, len(val_frames))
        if val_samples > 0:
            val_selected = random.sample(list(val_frames), val_samples)
            sample_selections.extend([("val", frame) for frame in val_selected])
        
        # Get samples from test set
        test_df = pd.read_csv(self.output_dir / "test" / "annotations.csv")
        test_frames = test_df["frame_number"].unique()
        test_samples = min(num_samples - train_samples - val_samples, len(test_frames))
        if test_samples > 0:
            test_selected = random.sample(list(test_frames), test_samples)
            sample_selections.extend([("test", frame) for frame in test_selected])
        
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
        for split, frame_num in sample_selections:
            # Get annotations for this frame
            if split == "train":
                annos_df = train_df
            elif split == "val":
                annos_df = val_df
            else:
                annos_df = test_df
                
            frame_annos = annos_df[annos_df["frame_number"] == frame_num]
            
            # Load image
            img_path = self.output_dir / split / "images" / f"image_{int(frame_num):06d}.jpg"
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
            
            # Save the visualization with split info in the filename
            vis_path = vis_dir / f"{split}_sample_{int(frame_num):06d}.jpg"
            cv2.imwrite(str(vis_path), img)
        
        print(f"Created {len(sample_selections)} visualization samples")
    
    def run(self):
        """Run the full dataset creation process."""
        self.find_all_images()
        self.create_dataset()
        self.create_visualizations()
        print("All done!")

# Create the dataset
if __name__ == "__main__":
    # Change the ratios here for your preferred split
    creator = BalancedDatasetCreator(target_size=500, train_ratio=0.7, val_ratio=0.15)
    creator.run()