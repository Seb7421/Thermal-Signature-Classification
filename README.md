# Thermal Signature Classification for Wildlife Conservation

This project demonstrates fine-tuning a convolutional neural network (CNN) to classify thermal signatures from infrared drone imagery as either human or animal. This classification is crucial for anti-poaching efforts and wildlife conservation.

## About the Project

The model uses transfer learning with a pre-trained ResNet-18 architecture to classify thermal signatures captured by drones in wildlife conservation areas. By accurately distinguishing between humans (potential poachers) and animals, conservation teams can allocate resources more efficiently in anti-poaching efforts.

## Dataset

This project uses a subset of the BirdsAI dataset:

- **Source**: [BirdsAI dataset](https://lila.science/datasets/conservationdrones)
- **Authors**: Bondi E, Jain R, Aggrawal P, Anand S, Hannaford R, Kapoor A, Piavis J, Shah S, Joppa L, Dilkina B, Tambe M.
- **Publication**: BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2020.
- **License**: Community Data License Agreement (permissive variant)

### Citation

If you use this code or the BirdsAI dataset, please cite:

```
@inproceedings{bondi2020birdsai,
  title={BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos},
  author={Bondi, Elizabeth and Jain, Raghav and Aggrawal, Palash and Anand, Saket and Hannaford, Robert and Kapoor, Ashish and Piavis, Jim and Shah, Shital and Joppa, Lucas and Dilkina, Bistra and Tambe, Milind},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2020}
}
```

## Data Preparation

This project includes a data preparation script (`data_preperation/prepare_data.py`) that helps create a balanced dataset for training, validation, and testing. The script:

1. Extracts frames with annotations from thermal videos
2. Processes annotations to ensure proper format
3. Splits data into train, validation, and test sets
4. Balances classes to ensure equal representation of humans and animals

### Using the Data Preparation Script

1. **Setup the environment**:
   ```bash
   conda env create -f data_preperation/data_preperation_env.yml
   conda activate birdsai-cnn
   ```

2. **Prepare your directory structure**:
   The script expects the following directory structure:
   ```
   data_preperation/
   ├── human_annotations/
   ├── unkown_animal_annotations/
   ├── elephant_annotations/
   ├── human_images/
   ├── unknown_animal_images/
   └── elephant_images/
   ```

3. **Download additional data (optional)**:
   You can download more image sequences and annotations from the [BirdsAI dataset](https://lila.science/datasets/conservationdrones) to enhance your dataset. Add the downloaded files to their respective directories:
   - Place human video sequences in `human_images/`
   - Place animal video sequences in `unknown_animal_images/` or `elephant_images/`
   - Place human annotations in `human_annotations/`
   - Place animal annotations in `unkown_animal_annotations/` or `elephant_annotations/`

4. **Run the script**:
   ```bash
   cd data_preperation
   python prepare_data.py
   ```

5. **Customize dataset parameters (optional)**:
   You can modify the following parameters in the script:
   - `target_size`: Total number of images for the dataset
   - `train_ratio`: Proportion of images for training
   - `val_ratio`: Proportion of images for validation
   ```python
   creator = FrameBasedDatasetCreator(target_size=700, train_ratio=0.7, val_ratio=0.15)
   ```

The script will create a balanced dataset in a directory called `birdsai_data/` with the structure required for the CNN training notebook.

## Features

- Thermal signature extraction from infrared drone imagery
- Data balancing to address class imbalance
- Transfer learning with ResNet-18
- Hyperparameter exploration (learning rate, freezing strategies)
- Comprehensive evaluation with confusion matrices and visualizations

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- OpenCV
- PIL (Pillow)

## Usage

The main code is contained in the Jupyter notebook `thermal_signature_classification.ipynb`. To run the code:

1. Clone this repository
2. If you don't have the dataset ready:
   - Set up and run the data preparation script (see Data Preparation section)
   - Or download the required subset of the BirdsAI dataset (contact the authors for access)
3. Place the dataset in a directory named `birdsai_data` with the expected structure
4. Set up the CNN environment: `conda env create -f cnn_notebook_env.yml`
5. Activate the environment: `conda activate cnn-tutorial`
6. Run the Jupyter notebook: `jupyter notebook thermal_signature_classification.ipynb`

## Project Structure

```
.
├── LICENSE.md
├── README.md
├── thermal_signature_classification.ipynb
├── cnn_notebook_env.yml
├── data_preperation/
│   ├── data_preperation_env.yml
│   └── prepare_data.py
└── birdsai_data/
    ├── metadata.json
    ├── train/
    │   ├── images/
    │   └── annotations.csv
    ├── val/
    │   ├── images/
    │   └── annotations.csv
    └── test/
        ├── images/
        └── annotations.csv
```

## Acknowledgements

This project acknowledges the BirdsAI dataset, which was supported by Microsoft AI for Earth, NSF grants CCF-1522054 and IIS-1850477, MURI W911NF-17-1-0370, and the Infosys Center for Artificial Intelligence, IIIT-Delhi.

For questions about the dataset, contact Elizabeth Bondi at Harvard University (ebondi@g.harvard.edu).

## License

This project is licensed under the Community Data License Agreement (permissive variant) - see the [LICENSE.md](LICENSE.md) file for details.
```

To use this content:
1. Copy the entire text block above (including the triple backticks)
2. Paste it into your text editor
3. Remove the first and last lines that contain the triple backticks
4. Save the file as README.md

The content includes proper Markdown formatting with headings, bullet points, code blocks, and links, ready to be displayed correctly on GitHub or other platforms that support Markdown.