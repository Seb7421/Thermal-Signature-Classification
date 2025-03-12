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
2. Download the required subset of the BirdsAI dataset (contact the authors for access)
3. Place the dataset in a directory named `birdsai_data` with the expected structure
4. Run the Jupyter notebook

## Project Structure

```
.
├── LICENSE.md
├── README.md
├── thermal_signature_classification.ipynb
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