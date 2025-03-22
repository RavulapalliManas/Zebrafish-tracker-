# Lightweight tracking software for analyzing behavior of Single adult Zebrafish 

- Can also be used for larvae but need to ensure better lighting conditions and clear separation between fish and background

## Experimental setup for ideal results
- Tank with good illumination and clear separation between fish and background
- Camera should cover the entire tank and if possible use a ND/ polarizing filter to reduce glare and reflections
- Ideally camera should be at-least 1080p 60fps but we can work with lower resolutions namely 720p 30fps
- Fix the camera strongly so that it doesn't shake during recording and ensure no moving objects within frame other than fish
- Fish size must be medium small to large size for ideal tracking if the fish is too small ensure that water in tank isn't cloudy
- Ensure that camera lens and tank glass are clean and free from dust and smudges as it may affect the tracking


## Usage
- # Running the Fish Tracking Software

To use the fish tracking software, follow these simple steps in your terminal:

## Basic Usage

1.  Navigate to the project directory:
    ```bash
    cd path/to/Fish-tracking
    ```

2.  Install dependencies (if you haven't already):
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the main program:
    ```bash
    python Main.py
    ```

4.  Follow the interactive prompts:
    * When asked for processing mode, type `'s'` for single video or `'b'` for batch processing.
    * When asked for file source, type `'g'` for Google Drive, `'o'` for OneDrive, or `'i'` for local files.
    * Enter the output directory path or press Enter to use the default (`./Data`).

## Single Video Example

```bash
$ python Main.py
Fish Tracking System
===================
Select processing mode: (s)ingle video or (b)atch processing? (s/b): s
Select file source: (g)oogle Drive, (o)neDrive, or (i)nternal files? (g/o/i): i
Enter output directory path (default: ./Data): ./MyResults
Enter path to video file: /path/to/my_fish_video.mp4
Enable visualization? (y/n): y
```

## Background on how processing is happening

# Fish Detection and Tracking: Technical Overview

This document explains how the fish detection and tracking process works in our Fish-tracking system.

## 1. Image Preprocessing

Before attempting to detect fish, each video frame undergoes several preprocessing steps:

### 1.1 Scaling and Denoising

-   Frames are optionally resized (scaled) to improve processing speed.
-   Fast Non-Local Means denoising algorithm removes noise while preserving edges.
-   Parameters: h=10 (filter strength for luminance), hColor=10 (color components), templateWindowSize=7, searchWindowSize=21.

### 1.2 Color and Contrast Enhancement

-   Multi-Scale Retinex with Color Restoration (MSRCR) algorithm enhances image details in varying illumination.
-   Three scales (sigma values of 15, 80, 250) are used to capture details at different frequencies.
-   LAB color space enhancement separates luminance from color, improving contrast without color distortion.
-   CLAHE (Contrast Limited Adaptive Histogram Equalization) further improves local contrast.

### 1.3 Edge-preserving Smoothing

-   Bilateral filtering smooths the image while preserving important edges.
-   Parameters: d=9 (diameter of each pixel neighborhood), sigmaColor=75, sigmaSpace=75.

## 2. Background Subtraction

### 2.1 KNN Background Subtractor

-   K-Nearest Neighbors background subtractor separates moving objects (foreground) from static background.
-   Parameters: history=1000 (number of frames to build background model), dist2Threshold=400 (threshold on squared distance).
-   Shadow detection is enabled to differentiate shadows from actual fish.

### 2.2 Post-processing of Foreground Mask

-   Binary thresholding removes shadow pixels (typically marked as 127 in mask).
-   Median blur filters out small noise in the mask.
-   Morphological operations (closing followed by opening) with elliptical kernel smooth the mask.

## 3. Fish Detection and Validation

### 3.1 Contour Detection

-   Canny edge detection identifies boundaries in the processed mask.
-   Contours are extracted from these edges using `cv2.findContours`.

### 3.2 Contour Filtering

-   Area-based filtering: Only contours with area between `MIN_FISH_AREA` (100) and `MAX_FISH_AREA` (2000) pixels are kept.
-   Shape validation through multiple metrics:
    * a) Aspect ratio: Width-to-height ratio must be between 0.3 and 3.0 (fish are typically elongated).
    * b) Solidity: Ratio of contour area to its convex hull area (must be between 0.5 and 0.95).
    * c) Circularity: 4π × Area / (Perimeter²) (must be between 0.2 and 0.8).

### 3.3 Reflection Detection

-   Checks RGB values at the detected center point.
-   If all RGB values are close to each other (within tolerance of 10), it's flagged as a potential reflection.
-   When reflection is detected, the system uses the previous valid detection instead.

## 4. Fish Tracking

### 4.1 Movement Validation

-   Tracks fish movement between frames by comparing current and previous positions.
-   Maximum distance threshold (200 pixels) prevents unrealistic jumps.
-   If distance exceeds threshold, previous valid position is used instead.

### 4.2 Position History

-   Maintains a window of recent positions (typically 1 second worth of frames).
-   Used to calculate cumulative speed over time, providing more stable measurements.

### 4.3 Box Intersection Detection

-   Determines which defined box (area of interest) contains the fish.
-   Uses point-in-polygon test to check if fish center is inside a box.
-   Time spent and distance traveled are tracked for each box.

## 5. Speed and Distance Calculation

### 5.1 Distance Calculation

-   Euclidean distance between consecutive positions: √[(x₂-x₁)² + (y₂-y₁)²].
-   Pixel distances are converted to meters using `PIXEL_TO_METER` conversion factor (0.000099 m/pixel).

### 5.2 Speed Calculation

-   Instantaneous speed: distance / time between frames.
-   Average speed in box: total distance in box / total time in box.
-   Cumulative speed: distance over a window of frames / time window (provides more stable measurement).

## 6. Tank Boundary Handling

### 6.1 Tank Mask Creation

-   User defines tank boundary by clicking points on first frame