# 🏙️ What Makes Amherst Look Like Amherst?

> **COMPSCI 682 - Computer Vision Project**  
> University of Massachusetts Amherst  
> By [Preston Yee](https://github.com/PYee1999) & [Mehak Virender Nargotra](mailto:mnargotra@umass.edu)

This project explores the distinct visual identity of Amherst, MA using modern deep learning techniques like ResNet classification and Class Activation Mapping (CAM). We aimed to answer:  
**What makes Amherst look like Amherst?**

---

## 💡 Project Overview

We trained a deep learning pipeline on images from **Amherst** and **19+ other cities** using:

- Binary and Multiclass ResNet-based classifiers  
- CAM, GradCAM, and SmoothGradCAM visualizations  
- Custom datasets from YouTube walking tours and original Amherst footage  
- PyTorch and Torchvision for deep learning and image transformation

![Sample cities](images/page3_img1.jpeg)  
<sub>Sample images collected from cities like Amherst, Amsterdam, Venice, and more</sub>

---

## 🧠 Methodology

### Dataset Collection
- 10,000 images of Amherst (UMass campus + town)
- 5,000+ images from each of 19 other cities (Venice, Pune, Amsterdam, etc.)
- Images were extracted from walking tour videos using FFmpeg

### Model Design
- **Base model:** ResNet-50 with pretrained weights
- Binary classification: Amherst vs. Non-Amherst
- Multiclass classification: 20-way city classifier
- Loss functions: `BinaryCrossEntropyLoss`, `CrossEntropyLoss`
- Optimizers: `Adam`, `SGD` with 5/10 epochs

### Visualization
- CAM techniques applied to validation samples
- Analyzed which regions of Amherst images contributed to classification
- Focus on architectural features, building layouts, and patterns

---

## 🚀 How to Run

1. **Download the Dataset**  
   [📥 Download Link (Google Drive)](https://drive.google.com/file/d/1llG0ntOjmZED_qRNqmnXFgRIUY-E7ALy/view?usp=sharing)

2. **Unzip the data**  
   You will get `Binary_Dataset` and `Nonbinary_Dataset`.

3. **Set paths in `code/run.py`**
   ```python
   BINARY_DATASET_PATH = "your/path/to/Binary_Dataset"
   NONBINARY_DATASET_PATH = "your/path/to/Nonbinary_Dataset"

---

## 📊 Key Results

The best performing model used **binary classification**, the **Adam optimizer**, and **5 training epochs**. This configuration achieved a **validation accuracy of 68%** on Amherst images. Class Activation Maps (CAMs) clearly highlighted visual elements unique to Amherst, including:

- Prominent campus buildings such as the **Du Bois Library**
- Distinct architectural layouts and spacing
- Walkways and angled roof structures typical of UMass Amherst

![CAM - Best Accuracy](images/page7_img7.png)  
<sub>Validation accuracy over training epochs for different configurations — best result achieved with Binary + Adam optimizer</sub>

---

## 🔍 Additional Visualizations

We also explored **SmoothGradCAM**, which yielded more focused attention maps. These visualizations captured fine-grained features such as:

- Rooftop shapes
- Window placements
- Subtle structural patterns

![SmoothGradCAM](images/page7_img9.png)  
<sub>Localized attention to architectural details using SmoothGradCAM</sub>

---

## 🧩 Future Work

- Integrate **Vision Transformers (ViTs)** to improve feature localization
- Apply **semantic segmentation** to identify buildings, foliage, and walkways
- Improve dataset diversity to include more rural and suburban comparisons
- Extend from binary classification to **object-level tagging and labeling**

---

## 👨‍💻 Developers

- **Mehak Virender Nargotra** — [mnargotra@umass.edu](mailto:mnargotra@umass.edu)  
- **Preston Yee** — [GitHub](https://github.com/PYee1999)

---

## 📎 Acknowledgements

- Inspired by [_What Makes Paris Look Like Paris?_ (Doersch et al., SIGGRAPH 2012)](https://graphics.cs.cmu.edu/projects/what-makes-paris/paris.pdf)  
- Visualization techniques based on **Grad-CAM** and **SmoothGradCAM++**  
- Built using tools like **PyTorch**, **Torchvision**, **TorchCAM**, **FFmpeg**, and **YouTube-DL**
