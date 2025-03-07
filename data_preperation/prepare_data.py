import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import random
import json
import shutil
from tqdm import tqdm
from collections import defaultdict

class FrameBasedDatasetCreator:
    def __init__(self, base_dir=".", target_size=500, train_ratio=0.7, val_ratio=0.15):
        """
        Create a balanced dataset for human vs animal classification with frame-level splitting.
        
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
        
        # Output paths
        self.output_dir = self.base_dir / "birdsai_data"
        (self.output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "images").mkdir(parents=True, exist_ok=True)
        
        # Column names for the CSV files
        self.columns = [
            "frame_number", "object_id", "x", "y", "w", "h", 
            "class", "species", "occlusion", "noise"
        ]
        
        # For collecting all frames
        self.human_frames = []
        self.animal_frames = []
        
        # For statistics
        self.stats = {
            "total_found": 0,
            "human_found": 0,
            "animal_found": 0,
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
    
    def find_all_frames(self):
        """Find all images with annotations and collect them by class."""
        print("Finding all annotated frames...")
        
        # Process human annotations
        self._process_annotation_dir(
            self.human_anno_dir, 
            self.human_img_dir, 
            self.human_frames, 
            is_human=True
        )
        
        # Process unknown animal annotations
        self._process_annotation_dir(
            self.unknown_anno_dir, 
            self.unknown_img_dir, 
            self.animal_frames, 
            is_human=False, 
            source="unknown_animal"
        )
        
        # Process elephant annotations
        self._process_annotation_dir(
            self.elephant_anno_dir, 
            self.elephant_img_dir, 
            self.animal_frames, 
            is_human=False, 
            source="elephant"
        )
        
        # Update statistics
        self.stats["human_found"] = len(self.human_frames)
        self.stats["animal_found"] = len(self.animal_frames)
        self.stats["total_found"] = len(self.human_frames) + len(self.animal_frames)
        
        print(f"Found {self.stats['total_found']} annotated images:")
        print(f"  - Human: {self.stats['human_found']} images")
        print(f"  - Animal: {self.stats['animal_found']} images")
    
    def _process_annotation_dir(self, anno_dir, img_dir, frames_list, is_human=True, source="human"):
        """Process all annotation files in a directory and collect frames."""
        class_value = 1 if is_human else 0
        
        for anno_file in anno_dir.glob("*.csv"):
            sequence_id = anno_file.stem
            sequence_dir = img_dir / sequence_id
            
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
                        # Fix class values to ensure consistency
                        frame_df_fixed = frame_df.copy()
                        frame_df_fixed["class"] = class_value
                        
                        # Store this frame
                        frames_list.append({
                            "image_path": image_path,
                            "annotations": frame_df_fixed.to_dict("records"),
                            "source": source,
                            "sequence_id": sequence_id,
                            "frame_id": frame_id
                        })
    
    def create_dataset(self):
        """Create a balanced dataset with frame-level split into train, val and test."""
        print(f"Creating balanced dataset with frame-level splitting...")
        
        # Split frames into train, val, test ensuring class balance
        train_frames, val_frames, test_frames = self._split_frames()
        
        # Process the frames for each split
        self._process_split(train_frames, "train")
        self._process_split(val_frames, "val")
        self._process_split(test_frames, "test")
        
        # Update total statistics
        all_frames = train_frames + val_frames + test_frames
        total_selected = len(all_frames)
        
        # Calculate frame sources
        source_counts = defaultdict(int)
        for frame in all_frames:
            source_counts[frame["source"]] += 1
        
        # Calculate average annotations per image
        total_annotations = (self.stats["train_annotations"] + 
                             self.stats["val_annotations"] + 
                             self.stats["test_annotations"])
        avg_annotations = total_annotations / total_selected if total_selected > 0 else 0
        
        # Calculate sequence distribution across splits
        sequence_distribution = self._analyze_sequence_distribution(train_frames, val_frames, test_frames)
        
        # Save metadata
        metadata = {
            "target_size": self.target_size,
            "actual_dataset_size": total_selected,
            "train_images": len(train_frames),
            "val_images": len(val_frames),
            "test_images": len(test_frames),
            "human_images": sum(1 for frame in all_frames if frame["annotations"][0]["class"] == 1),
            "animal_images": sum(1 for frame in all_frames if frame["annotations"][0]["class"] == 0),
            
            "total_annotations": total_annotations,
            "train_annotations": self.stats["train_annotations"],
            "val_annotations": self.stats["val_annotations"],
            "test_annotations": self.stats["test_annotations"],
            "human_annotations": self.stats["human_annotations"],
            "animal_annotations": self.stats["animal_annotations"],
            "avg_annotations_per_image": round(avg_annotations, 2),
            
            "image_sources": dict(source_counts),
            "species_distribution": self.stats["species_distribution"],
            "sequence_distribution": sequence_distribution
        }
        
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("Dataset creation completed successfully!")
        print(f"Total images: {total_selected}")
        print(f"Total annotations: {total_annotations}")
        print(f"Average annotations per image: {avg_annotations:.2f}")
        
        # Print sequence distribution summary
        print("\nSequence Distribution Summary:")
        for seq_id, counts in sequence_distribution.items():
            print(f"  Sequence {seq_id}: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    
    def _analyze_sequence_distribution(self, train_frames, val_frames, test_frames):
        """Analyze how sequences are distributed across splits"""
        sequence_distribution = {}
        
        # Count frames from each sequence in each split
        for frame in train_frames:
            seq_id = frame["sequence_id"]
            if seq_id not in sequence_distribution:
                sequence_distribution[seq_id] = {"train": 0, "val": 0, "test": 0}
            sequence_distribution[seq_id]["train"] += 1
            
        for frame in val_frames:
            seq_id = frame["sequence_id"]
            if seq_id not in sequence_distribution:
                sequence_distribution[seq_id] = {"train": 0, "val": 0, "test": 0}
            sequence_distribution[seq_id]["val"] += 1
            
        for frame in test_frames:
            seq_id = frame["sequence_id"]
            if seq_id not in sequence_distribution:
                sequence_distribution[seq_id] = {"train": 0, "val": 0, "test": 0}
            sequence_distribution[seq_id]["test"] += 1
        
        return sequence_distribution
    
    def _split_frames(self):
        """Split frames into train, val, test ensuring class balance and diverse sequence representation"""
        print("Using frame-level splitting to ensure each sequence contributes to all splits...")
        
        # Group human frames by sequence
        human_by_sequence = defaultdict(list)
        for frame in self.human_frames:
            human_by_sequence[frame["sequence_id"]].append(frame)
        
        # Group animal frames by sequence
        animal_by_sequence = defaultdict(list)
        for frame in self.animal_frames:
            animal_by_sequence[frame["sequence_id"]].append(frame)
        
        # Initialize output splits
        train_frames = []
        val_frames = []
        test_frames = []
        
        # Calculate target sizes for each split
        total_size = min(self.target_size, len(self.human_frames) + len(self.animal_frames))
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size
        
        # Set target sizes for each class in each split
        # Aim for roughly 50/50 distribution between humans and animals
        human_train_target = min(train_size // 2, len(self.human_frames))
        animal_train_target = train_size - human_train_target
        
        human_val_target = min(val_size // 2, len(self.human_frames) - human_train_target)
        animal_val_target = val_size - human_val_target
        
        human_test_target = min(test_size // 2, len(self.human_frames) - human_train_target - human_val_target)
        animal_test_target = test_size - human_test_target
        
        print(f"Target split sizes:")
        print(f"  Train: {train_size} images ({human_train_target} human, {animal_train_target} animal)")
        print(f"  Val: {val_size} images ({human_val_target} human, {animal_val_target} animal)")
        print(f"  Test: {test_size} images ({human_test_target} human, {animal_test_target} animal)")
        
        # Function to split frames from one sequence across train/val/test
        def split_sequence_frames(frames, train_ratio=0.7, val_ratio=0.15):
            random.shuffle(frames)  # Randomize frames from this sequence
            n = len(frames)
            train_idx = int(n * train_ratio)
            val_idx = int(n * (train_ratio + val_ratio))
            
            return frames[:train_idx], frames[train_idx:val_idx], frames[val_idx:]
        
        # Process human sequences
        human_train, human_val, human_test = [], [], []
        
        for seq_id, frames in human_by_sequence.items():
            seq_train, seq_val, seq_test = split_sequence_frames(frames, self.train_ratio, self.val_ratio)
            human_train.extend(seq_train)
            human_val.extend(seq_val)
            human_test.extend(seq_test)
        
        # Process animal sequences
        animal_train, animal_val, animal_test = [], [], []
        
        for seq_id, frames in animal_by_sequence.items():
            seq_train, seq_val, seq_test = split_sequence_frames(frames, self.train_ratio, self.val_ratio)
            animal_train.extend(seq_train)
            animal_val.extend(seq_val)
            animal_test.extend(seq_test)
        
        # Sample from each class to meet target sizes
        def sample_frames(frames, target_size):
            if len(frames) <= target_size:
                return frames
            return random.sample(frames, target_size)
        
        human_train = sample_frames(human_train, human_train_target)
        animal_train = sample_frames(animal_train, animal_train_target)
        
        human_val = sample_frames(human_val, human_val_target)
        animal_val = sample_frames(animal_val, animal_val_target)
        
        human_test = sample_frames(human_test, human_test_target)
        animal_test = sample_frames(animal_test, animal_test_target)
        
        # Combine and shuffle each split
        train_frames = human_train + animal_train
        val_frames = human_val + animal_val
        test_frames = human_test + animal_test
        
        random.shuffle(train_frames)
        random.shuffle(val_frames)
        random.shuffle(test_frames)
        
        # Log the split statistics
        print(f"Split frames:")
        train_humans = sum(1 for frame in train_frames if frame["annotations"][0]["class"] == 1)
        train_animals = len(train_frames) - train_humans
        
        val_humans = sum(1 for frame in val_frames if frame["annotations"][0]["class"] == 1)
        val_animals = len(val_frames) - val_humans
        
        test_humans = sum(1 for frame in test_frames if frame["annotations"][0]["class"] == 1)
        test_animals = len(test_frames) - test_humans
        
        print(f"  - Train: {len(train_frames)} images ({train_humans} human, {train_animals} animal)")
        print(f"  - Validation: {len(val_frames)} images ({val_humans} human, {val_animals} animal)")
        print(f"  - Test: {len(test_frames)} images ({test_humans} human, {test_animals} animal)")
        
        # Count sequences in each split
        train_sequences = set(frame["sequence_id"] for frame in train_frames)
        val_sequences = set(frame["sequence_id"] for frame in val_frames)
        test_sequences = set(frame["sequence_id"] for frame in test_frames)
        
        print(f"Sequence coverage:")
        print(f"  - Train: {len(train_sequences)} sequences")
        print(f"  - Validation: {len(val_sequences)} sequences")
        print(f"  - Test: {len(test_sequences)} sequences")
        
        # Calculate sequence overlap
        train_val_overlap = len(train_sequences.intersection(val_sequences))
        train_test_overlap = len(train_sequences.intersection(test_sequences))
        val_test_overlap = len(val_sequences.intersection(test_sequences))
        all_overlap = len(train_sequences.intersection(val_sequences).intersection(test_sequences))
        
        print(f"Sequence overlap:")
        print(f"  - Train/Val overlap: {train_val_overlap} sequences")
        print(f"  - Train/Test overlap: {train_test_overlap} sequences")
        print(f"  - Val/Test overlap: {val_test_overlap} sequences")
        print(f"  - All splits overlap: {all_overlap} sequences")
        
        return train_frames, val_frames, test_frames
    
    def _process_split(self, frames, split_name):
        """Process a split of frames and annotations."""
        annotations = []
        for i, frame_data in enumerate(tqdm(frames, desc=f"Processing {split_name} images")):
            # Create output image name and path
            img_filename = f"image_{i:06d}.jpg"
            img_output_path = self.output_dir / split_name / "images" / img_filename
            
            # Copy the image
            img = cv2.imread(str(frame_data["image_path"]))
            cv2.imwrite(str(img_output_path), img)
            
            # Process annotations
            for anno in frame_data["annotations"]:
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
        
        print(f"Processed {split_name} split: {len(frames)} images, {len(annotations)} annotations")
    
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
        self.find_all_frames()
        self.create_dataset()
        self.create_visualizations()
        print("All done!")

# Create the dataset
if __name__ == "__main__":
    # Change the ratios here for your preferred split
    creator = FrameBasedDatasetCreator(target_size=500, train_ratio=0.7, val_ratio=0.15)
    creator.run()