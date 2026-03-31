Advanced Image Enhancement & Feature Analysis Pipeline
======================================================

**Course:** CSE3010 – Computer Vision

📌 Overview
-----------

This project implements a **complete classical computer vision pipeline** designed to improve image quality and extract meaningful features from images.

Unlike deep learning approaches, this pipeline relies entirely on **mathematical models and signal processing techniques**. It demonstrates how traditional CV methods can effectively handle:

*   Noise removal
    
*   Contrast enhancement
    
*   Detail restoration
    
*   Feature detection
    

The final output provides both **visual results** and **quantitative metrics**, making it suitable for analysis and academic evaluation.

⚙️ Pipeline Stages (Detailed Explanation)
-----------------------------------------

### 1️⃣ Preprocessing (Noise Reduction)

**Goal:** Remove unwanted noise while preserving important structures.

#### 🔹 Gaussian Blur

*   Applies a weighted average using a Gaussian function.
    
*   Smooths the image by reducing **high-frequency noise** (random variations).
    
*   Helps in stabilizing later stages like edge detection.
    

👉 Think of it as: _softly blurring the image to remove grainy noise._

#### 🔹 Median Filter

*   Replaces each pixel with the **median of its neighborhood**.
    
*   Especially effective for **salt-and-pepper noise**.
    
*   Unlike averaging, it **preserves edges**.
    

👉 Think of it as: _removing sudden spikes (black/white dots) without blurring edges._

### 2️⃣ Enhancement (Contrast Improvement)

**Goal:** Improve visibility of details, especially in low-contrast regions.

#### 🔹 CLAHE (Contrast Limited Adaptive Histogram Equalization)

*   Works on small regions (tiles) instead of the whole image.
    
*   Enhances local contrast rather than global brightness.
    
*   Uses a **clip limit** to avoid over-amplifying noise.
    

#### Why LAB Color Space?

*   Only the **L (lightness)** channel is modified.
    
*   Prevents distortion of colors (important for realistic output).
    

👉 Think of it as: _making dark areas clearer without ruining colors._

### 3️⃣ Restoration (Sharpening & Detail Recovery)

**Goal:** Enhance edges and fine details after smoothing.

#### 🔹 Unsharp Masking

Formula:

`   Sharpened = Original × (1 + α) − Blurred × α   `

*   Extracts edges by subtracting a blurred version.
    
*   Adds them back to amplify details.
    
*   Controlled using a **strength parameter (α)**.
    

👉 Think of it as: _highlighting edges by boosting differences._

#### 🔹 Laplacian Sharpening (Alternative)

*   Uses second-order derivatives to detect rapid intensity changes.
    
*   Emphasizes edges strongly.
    
*   Can be slightly more aggressive than unsharp masking.
    

👉 Think of it as: _detecting and enhancing boundaries directly._

### 4️⃣ Feature Analysis (Understanding Image Structure)

#### 🔹 Harris Corner Detection

*   Detects points where intensity changes in **multiple directions**.
    
*   Corners are important for:
    
    *   Object recognition
        
    *   Image matching
        
    *   Tracking
        

✔ Example: corners of buildings, windows, text edges

👉 Output: Red points marking detected corners

#### 🔹 ORB (Oriented FAST + Rotated BRIEF)

A fast and efficient feature detector + descriptor:

*   **FAST** → detects keypoints (interest points)
    
*   **Orientation** → assigns rotation to make it invariant
    
*   **BRIEF** → creates binary descriptors for matching
    

✔ Works well for:

*   Image matching
    
*   Feature tracking
    
*   SLAM systems
    

👉 Output: Green circles showing detected keypoints

### 5️⃣ Edge Detection (Canny)

**Goal:** Extract clean and thin edges from the image.

#### 🔹 Canny Edge Detector

Steps:

1.  Noise reduction
    
2.  Gradient calculation
    
3.  Non-maximum suppression
    
4.  Hysteresis thresholding
    

✔ Produces:

*   Thin edges
    
*   Minimal noise
    
*   Strong structural boundaries
    

👉 Output: White edges on black background

🖼️ Output Visualization
------------------------

The pipeline generates a **3×3 comparison grid**:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML
`   [Original]     [Preprocessed]     [CLAHE]  
    [Sharpened]    [Harris]           [ORB]  
    [Canny]        [Histogram]   `

### 📊 Histogram Panel

*   Shows pixel intensity distribution before vs after CLAHE
    
*   After enhancement → histogram spreads out → better contrast
    

### 📈 Metrics Bar

Metric Meaning

PSNR Difference from original (informational only)

Laplacian Variance Sharpness measure

Harris Corners Number of corner regions

ORB Keypoints Number of detected features

Time Total processing time

📦 Requirements
---------------

*   Python 3.8+
    
*   OpenCV
    
*   NumPy
    

💻 Installation (Windows)
-------------------------

### Step 1: Install Python

Download from: [https://www.python.org/]

### Step 2: Install dependencies

`   pip install opencv-python numpy   `

### Step 3: Run the program

#### ✅ Demo mode

`   python image_enhancement_pipeline.py demo   `

#### ✅ Process your own image

`   python image_enhancement_pipeline.py process your_image.jpg   `

### Optional arguments

`   --sharpen laplacian  --clip 3  --tile 8  --save-stages  --no-display   `

☁️ Running on Google Colab
--------------------------

### Step 1: Install

`   !pip install opencv-python   `

### Step 2: Upload files

`   from google.colab import files  files.upload()   `

### Step 3: Run

`   !python image_enhancement_pipeline.py process test.jpg --no-display   `

### Step 4: Display

`   from IPython.display import Image  Image("test_enhanced.png")   `

🧪 Recommended Test Images
--------------------------

Use:

*   Buildings (best for corners)
    
*   Text documents
    
*   Street scenes
    

Avoid:

*   Blank or low-detail images
    
*   Extremely blurry images
    

🧠 How to Verify It Works
-------------------------

✔ Noise reduced in preprocessing
✔ Contrast improved after CLAHE
✔ Edges sharper after restoration
✔ Corners detected correctly
✔ ORB features visible
✔ Canny edges clean
✔ Histogram spreads after enhancement
![test](https://github.com/user-attachments/assets/4d1e5856-f41e-4383-b70c-023a353b1442)
<img width="1200" height="940" alt="test_enhanced" src="https://github.com/user-attachments/assets/dd43b15d-72c1-4f3a-89b1-3258065690a9" />


🚀 Features
-----------

*   Fully classical CV pipeline
    
*   Modular and extensible
    
*   CLI-based usage
    
*   Works on real and synthetic images
    
*   Robust against edge cases
    

📁 File Structure
-----------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML
`   ├── image_enhancement_pipeline.py `      
`   ├── README.md  `
`   ├── demo_output.png `
`   └── test_images/   `

🏁 Conclusion
-------------

This project shows that **classical computer vision techniques remain powerful and interpretable**, capable of handling complex image processing tasks without deep learning.

👤 Author
---------

Jhanavi Pidugu

📌 License
----------
MIT License
For academic use only (CSE3010 Computer Vision)
