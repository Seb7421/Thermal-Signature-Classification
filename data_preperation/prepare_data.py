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

class SequenceBasedDatasetCreator:
    def __init__(self, base_dir=".", target_size=500, train_ratio=0.7, val_ratio=0.15):
        """
        Create a balanced dataset for human vs animal classification with sequence-level splitting.
        
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
        self.output_dir = self.base_dir.parent / "birdsai_data"
        (self.output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "images").mkdir(parents=True, exist_ok=True)
        
        # Column names for the CSV files
        self.columns = [
            "frame_number", "object_id", "x", "y", "w", "h", 
            "class", "species", "occlusion", "noise"
        ]
        
        # For tracking sequences
        self.human_sequences = defaultdict(list)
        self.animal_sequences = defaultdict(list)
        
        # For statistics
        self.stats = {
            "total_found": 0,
            "human_found": 0,
            "animal_found": 0,
            "human_sequences": 0,
            "animal_sequences": 0,
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
    
    def find_all_sequences(self):
        """Find all images with annotations and group them by sequence."""
        print("Finding all annotated sequences and images...")
        
        # Process human annotations
        self._process_annotation_dir(
            self.human_anno_dir, 
            self.human_img_dir, 
            self.human_sequences, 
            is_human=True
        )
        
        # Process unknown animal annotations
        self._process_annotation_dir(
            self.unknown_anno_dir, 
            self.unknown_img_dir, 
            self.animal_sequences, 
            is_human=False, 
            source="unknown_animal"
        )
        
        # Process elephant annotations
        self._process_annotation_dir(
            self.elephant_anno_dir, 
            self.elephant_img_dir, 
            self.animal_sequences, 
            is_human=False, 
            source="elephant"
        )
        
        # Update statistics
        human_images = sum(len(frames) for frames in self.human_sequences.values())
        animal_images = sum(len(frames) for frames in self.animal_sequences.values())
        
        self.stats["human_found"] = human_images
        self.stats["animal_found"] = animal_images
        self.stats["total_found"] = human_images + animal_images
        self.stats["human_sequences"] = len(self.human_sequences)
        self.stats["animal_sequences"] = len(self.animal_sequences)
        
        print(f"Found {self.stats['total_found']} annotated images across {len(self.human_sequences) + len(self.animal_sequences)} sequences:")
        print(f"  - Human: {self.stats['human_found']} images in {self.stats['human_sequences']} sequences")
        print(f"  - Animal: {self.stats['animal_found']} images in {self.stats['animal_sequences']} sequences")
    
    def _process_annotation_dir(self, anno_dir, img_dir, sequence_dict, is_human=True, source="human"):
        """Process all annotation files in a directory and group by sequence."""
        class_value = 1 if is_human else 0
        
        for anno_file in anno_dir.glob("*.csv"):
            sequence_id = anno_file.stem
            sequence_dir = img_dir / sequence_id
            
            if not sequence_dir.exists():
                print(f"Warning: Directory not found for {sequence_id}")
                continue
                
            df = pd.read_csv(anno_file, header=None, names=self.columns)
            sequence_images = []
            
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
                        
                        # Store this image in its sequence
                        sequence_images.append({
                            "image_path": image_path,
                            "annotations": frame_df_fixed.to_dict("records"),
                            "source": source,
                            "sequence_id": sequence_id,
                            "frame_id": frame_id
                        })
            
            # Only add sequences that have at least some images
            if sequence_images:
                sequence_dict[sequence_id] = sequence_images
    
    def create_dataset(self):
        """Create a balanced dataset with sequence-level split into train, val and test."""
        print(f"Creating balanced dataset with sequence-level splitting...")
        
        # Step 1: Split sequences into train, val, test ensuring class balance
        train_sequences, val_sequences, test_sequences = self._split_sequences()
        
        # Step 2: Sample images from each split to reach target size
        # while maintaining class balance
        train_images, val_images, test_images = self._sample_images_from_splits(
            train_sequences, val_sequences, test_sequences
        )
        
        # Step 3: Process the images for each split
        self._process_split(train_images, "train")
        self._process_split(val_images, "val")
        self._process_split(test_images, "test")
        
        # Update total statistics
        all_images = train_images + val_images + test_images
        total_selected = len(all_images)
        
        # Calculate image sources
        source_counts = defaultdict(int)
        for img in all_images:
            source_counts[img["source"]] += 1
        
        # Calculate average annotations per image
        total_annotations = (self.stats["train_annotations"] + 
                             self.stats["val_annotations"] + 
                             self.stats["test_annotations"])
        avg_annotations = total_annotations / total_selected if total_selected > 0 else 0
        
        # Save metadata
        metadata = {
            "target_size": self.target_size,
            "actual_dataset_size": total_selected,
            "train_images": len(train_images),
            "val_images": len(val_images),
            "test_images": len(test_images),
            "human_images": sum(1 for img in all_images if img["annotations"][0]["class"] == 1),
            "animal_images": sum(1 for img in all_images if img["annotations"][0]["class"] == 0),
            
            "total_annotations": total_annotations,
            "train_annotations": self.stats["train_annotations"],
            "val_annotations": self.stats["val_annotations"],
            "test_annotations": self.stats["test_annotations"],
            "human_annotations": self.stats["human_annotations"],
            "animal_annotations": self.stats["animal_annotations"],
            "avg_annotations_per_image": round(avg_annotations, 2),
            
            "image_sources": dict(source_counts),
            "species_distribution": self.stats["species_distribution"]
        }
        
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("Dataset creation completed successfully!")
        print(f"Total images: {total_selected}")
        print(f"Total annotations: {total_annotations}")
        print(f"Average annotations per image: {avg_annotations:.2f}")
    
    def _split_sequences(self):
        """Split sequences into train, val, test ensuring class balance."""
        # First, split human sequences
        human_seq_ids = list(self.human_sequences.keys())
        random.shuffle(human_seq_ids)
        
        # Calculate splits for humans
        train_idx = int(len(human_seq_ids) * self.train_ratio)
        val_idx = int(len(human_seq_ids) * (self.train_ratio + self.val_ratio))
        
        human_train_seqs = {seq_id: self.human_sequences[seq_id] for seq_id in human_seq_ids[:train_idx]}
        human_val_seqs = {seq_id: self.human_sequences[seq_id] for seq_id in human_seq_ids[train_idx:val_idx]}
        human_test_seqs = {seq_id: self.human_sequences[seq_id] for seq_id in human_seq_ids[val_idx:]}
        
        # Then, split animal sequences
        animal_seq_ids = list(self.animal_sequences.keys())
        random.shuffle(animal_seq_ids)
        
        # Calculate splits for animals
        train_idx = int(len(animal_seq_ids) * self.train_ratio)
        val_idx = int(len(animal_seq_ids) * (self.train_ratio + self.val_ratio))
        
        animal_train_seqs = {seq_id: self.animal_sequences[seq_id] for seq_id in animal_seq_ids[:train_idx]}
        animal_val_seqs = {seq_id: self.animal_sequences[seq_id] for seq_id in animal_seq_ids[train_idx:val_idx]}
        animal_test_seqs = {seq_id: self.animal_sequences[seq_id] for seq_id in animal_seq_ids[val_idx:]}
        
        # Combine and return
        train_sequences = {**human_train_seqs, **animal_train_seqs}
        val_sequences = {**human_val_seqs, **animal_val_seqs}
        test_sequences = {**human_test_seqs, **animal_test_seqs}
        
        print(f"Split sequences:")
        print(f"  - Train: {len(train_sequences)} sequences ({len(human_train_seqs)} human, {len(animal_train_seqs)} animal)")
        print(f"  - Validation: {len(val_sequences)} sequences ({len(human_val_seqs)} human, {len(animal_val_seqs)} animal)")
        print(f"  - Test: {len(test_sequences)} sequences ({len(human_test_seqs)} human, {len(animal_test_seqs)} animal)")
        
        return train_sequences, val_sequences, test_sequences
    
    def _sample_images_from_splits(self, train_sequences, val_sequences, test_sequences):
        """Sample images from each split to achieve target size with class balance."""
        # First, calculate total sizes
        # Ensure we have at least some data in each split if possible
        min_size_per_split = 10
        
        # If we have less than the target size, adjust proportionally
        estimated_total_available = 0
        for seq_dict in [train_sequences, val_sequences, test_sequences]:
            for images in seq_dict.values():
                estimated_total_available += len(images)
        
        target_to_use = min(self.target_size, estimated_total_available)
        
        if target_to_use < 3 * min_size_per_split:
            # Not enough data for all splits, prioritize train
            if target_to_use < min_size_per_split:
                # Too little data, put everything in train
                total_train_size = target_to_use
                total_val_size = 0
                total_test_size = 0
            else:
                # Try to have at least some validation data
                total_train_size = target_to_use - min_size_per_split
                total_val_size = min_size_per_split
                total_test_size = 0
        else:
            # Normal case - use the ratios
            total_train_size = int(target_to_use * self.train_ratio)
            total_val_size = int(target_to_use * self.val_ratio)
            total_test_size = target_to_use - total_train_size - total_val_size
        
        # Get all images from each split
        train_human_images = []
        train_animal_images = []
        for seq_images in train_sequences.values():
            if seq_images and seq_images[0]["annotations"][0]["class"] == 1:
                train_human_images.extend(seq_images)
            else:
                train_animal_images.extend(seq_images)
        
        val_human_images = []
        val_animal_images = []
        for seq_images in val_sequences.values():
            if seq_images and seq_images[0]["annotations"][0]["class"] == 1:
                val_human_images.extend(seq_images)
            else:
                val_animal_images.extend(seq_images)
        
        test_human_images = []
        test_animal_images = []
        for seq_images in test_sequences.values():
            if seq_images and seq_images[0]["annotations"][0]["class"] == 1:
                test_human_images.extend(seq_images)
            else:
                test_animal_images.extend(seq_images)
        
        # Sample balanced subsets for each split
        train_images = self._sample_balanced_subset(
            train_human_images, train_animal_images, total_train_size
        )
        
        val_images = self._sample_balanced_subset(
            val_human_images, val_animal_images, total_val_size
        )
        
        test_images = self._sample_balanced_subset(
            test_human_images, test_animal_images, total_test_size
        )
        
        # Update statistics
        self.stats["train_images"] = len(train_images)
        self.stats["val_images"] = len(val_images)
        self.stats["test_images"] = len(test_images)
        
        # Log the split statistics
        print(f"Sampled images:")
        train_humans = sum(1 for img in train_images if img["annotations"][0]["class"] == 1)
        train_animals = len(train_images) - train_humans
        
        val_humans = sum(1 for img in val_images if img["annotations"][0]["class"] == 1)
        val_animals = len(val_images) - val_humans
        
        test_humans = sum(1 for img in test_images if img["annotations"][0]["class"] == 1)
        test_animals = len(test_images) - test_humans
        
        print(f"  - Train: {len(train_images)} images ({train_humans} human, {train_animals} animal)")
        print(f"  - Validation: {len(val_images)} images ({val_humans} human, {val_animals} animal)")
        print(f"  - Test: {len(test_images)} images ({test_humans} human, {test_animals} animal)")
        
        return train_images, val_images, test_images
    
    def _sample_balanced_subset(self, human_images, animal_images, target_size):
        """Sample a balanced subset of images with the specified target size."""
        if target_size <= 0:
            return []
        
        # Aim for a balanced split but handle cases where one class has fewer images
        half_target = target_size // 2
        humans_available = len(human_images)
        animals_available = len(animal_images)
        
        human_sample_size = min(half_target, humans_available)
        animal_sample_size = min(target_size - human_sample_size, animals_available)
        
        # If we don't have enough animals, use more humans if available
        if animal_sample_size < (target_size - human_sample_size):
            human_sample_size = min(humans_available, target_size - animal_sample_size)
        
        # Sample and combine
        sampled_humans = random.sample(human_images, human_sample_size) if human_sample_size > 0 else []
        sampled_animals = random.sample(animal_images, animal_sample_size) if animal_sample_size > 0 else []
        
        combined = sampled_humans + sampled_animals
        random.shuffle(combined)
        
        return combined
    
    def _process_split(self, images, split_name):
        """Process a split of images and annotations."""
        # Check if we have any images for this split
        if not images:
            print(f"Warning: No images for {split_name} split")
            # Create an empty annotations file to prevent errors
            annotations_df = pd.DataFrame(columns=self.columns)
            annotations_df.to_csv(self.output_dir / split_name / "annotations.csv", index=False)
            return
            
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
        
        # Helper to safely read CSV and handle empty files
        def safe_read_csv(file_path):
            try:
                if file_path.exists() and file_path.stat().st_size > 0:
                    return pd.read_csv(file_path)
                else:
                    print(f"Warning: File {file_path} is empty or doesn't exist.")
                    return pd.DataFrame()
            except pd.errors.EmptyDataError:
                print(f"Warning: No data found in {file_path}")
                return pd.DataFrame()
        
        # Get samples from train set
        train_file = self.output_dir / "train" / "annotations.csv"
        train_df = safe_read_csv(train_file)
        train_frames = train_df["frame_number"].unique() if not train_df.empty else []
        train_samples = min(num_samples // 3, len(train_frames))
        if train_samples > 0:
            train_selected = random.sample(list(train_frames), train_samples)
            sample_selections.extend([("train", frame) for frame in train_selected])
        
        # Get samples from val set
        val_file = self.output_dir / "val" / "annotations.csv"
        val_df = safe_read_csv(val_file)
        val_frames = val_df["frame_number"].unique() if not val_df.empty else []
        val_samples = min(num_samples // 3, len(val_frames))
        if val_samples > 0:
            val_selected = random.sample(list(val_frames), val_samples)
            sample_selections.extend([("val", frame) for frame in val_selected])
        
        # Get samples from test set
        test_file = self.output_dir / "test" / "annotations.csv"
        test_df = safe_read_csv(test_file)
        test_frames = test_df["frame_number"].unique() if not test_df.empty else []
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
            
            # Skip if empty dataframe
            if annos_df.empty:
                continue
                
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
        self.find_all_sequences()
        self.create_dataset()
        self.create_visualizations()
        print("All done!")

# Create the dataset
if __name__ == "__main__":
    # Change the ratios here for your preferred split
    creator = SequenceBasedDatasetCreator(target_size=500, train_ratio=0.7, val_ratio=0.15)
    creator.run()