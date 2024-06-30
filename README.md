# Automate Images Analyzer

<div>
    <a href="https://colab.research.google.com/drive/1_-MDGg3LWcWmeaQWYuzFBXlmTNrBUBWX?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  </div>

## Introduction
This project is designed to recognize beer brands from images using Object Detection and Optical Character Recognition (OCR) techniques. The project employs pre-trained models and various image processing libraries to analyze images and identify the presence of specific beer brands.

## Features
- Recognizes beer brands such as Heineken, Tiger, Bia Viet, Larue, Bivina, Edelweiss, and Strongbow.
- Detects the presence of people in the images.
- Generates image captions using the Salesforce BLIP model.
- Provides detailed insights and business problem analyses based on the images.

## Requirements
- Python 3.x
- Streamlit
- EasyOCR
- OpenCV
- Transformers
- Groq
- Levenshtein
- NumPy
- Torchvision
- PIL
- Localtunnel (for public URL generation)

## Installation
1. Install required Python packages:
    ```sh
    pip install opencv-python transformers groq python-Levenshtein streamlit numpy easyocr
    ```
2. Install Localtunnel:
    ```sh
    npm install -g localtunnel
    ```

3. (Optional) If you are running this on a Unix-based system, install zip/unzip tools:
    ```sh
    apt-get install zip unzip
    ```

## Usage
1. Clone this repository and navigate to the project directory.
2. Save the script provided in the project as `app.py`.
3. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```
4. Use Localtunnel to expose the Streamlit app to the public:
    ```sh
    lt --port 8501
    ```

## How to Use
1. Upload a zip file containing images via the Streamlit interface.
2. The app will process the images, classify them as branded or non-branded, and display the results.
3. You can view detailed insights for each image by selecting the appropriate option.

## Project Structure
- `app.py`: The main script containing all functionalities including image processing, OCR, object detection, and the Streamlit interface.
- `temp_folder/`: Temporary directory where uploaded images will be extracted and processed.

## Image Analysis
The project performs the following steps to analyze the images:
1. **OCR with EasyOCR**: Detects and extracts text from images.
2. **Image Captioning**: Generates captions for the images using the Salesforce BLIP model.
3. **Person Detection**: Uses Faster R-CNN model to detect people in the images.
4. **Brand Recognition**: Filters recognized texts to identify beer brands based on predefined keywords.
5. **Insight Generation**: Creates a detailed prompt and uses the Groq API to generate insights based on the image analysis.

## Business Problems Addressed
1. **Counting People Drinking Beer**: Identifies the number of people drinking Heineken beer.
2. **Detecting Promotional Materials**: Detects promotional materials with the Heineken logo.
3. **Evaluating Event Success**: Assesses the success of events by analyzing the number of attendees and their mood.
4. **Monitoring Marketing Staff**: Ensures the presence of at least two marketing staff members at each location.
5. **Evaluating Store Presence**: Assesses the quality of Heineken's presence in stores.

## Example Usage
Upload a zip file with images and let the app process them. The app will classify the images, detect brands, and provide insights based on the analysis.

## Contributors
- Vu Van Nam
- Tao Duc Minh
- Thai Gia Huy
