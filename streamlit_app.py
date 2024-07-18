import streamlit as st
import easyocr
import cv2
import Levenshtein
import os
from transformers import pipeline
from PIL import Image
import zipfile
from groq import Groq
import numpy as np


# Khởi tạo đối tượng OCR
reader = easyocr.Reader(['en'])

# Danh sách từ khóa thương hiệu bia
brand_keywords = {
    "Heineken": ["heineken", "coolpack"],
    "Tiger": ["tiger", "crystal", "khai xuan", "ban linh"],
    "Bia Viet": ["bia viet"],
    "Larue": ["larue", "special", "smooth"],
    "Bivina": ["bivina", "export"],
    "Edelweiss": ["edelweiss"],
    "Strongbow": ["strongbow"]
}

# Tiền xử lý ảnh
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    noise_removed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return noise_removed

# Nhận diện văn bản từ ảnh sử dụng EasyOCR
def detect_text(image_path):
    preprocessed_image = preprocess_image(image_path)
    result = reader.readtext(preprocessed_image)
    return result

# Lọc các từ liên quan đến thương hiệu
def filter_related_words(detected_texts, brand_keywords, threshold=2):
    filtered_texts = []
    for text, prob in detected_texts:
        for brand, keywords in brand_keywords.items():
            for keyword in keywords:
                if (Levenshtein.distance(text.lower(), keyword.lower()) <= threshold):
                    filtered_texts.append((text, prob, brand))
                    break
    return filtered_texts

# Phân biệt tất cả các thương hiệu bia dựa trên văn bản nhận diện
def identify_all_beer_brands(detected_texts, brand_keywords):
    found_brands = set()
    for text, prob, brand in detected_texts:
        found_brands.add(brand)
    return list(found_brands)

# Sử dụng pipeline của transformers để tạo tiêu đề cho hình ảnh
caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Hàm nhận diện logo thương hiệu
def detect_logos(image_path):
    detected_texts = detect_text(image_path)
    filtered_texts = filter_related_words([(text, prob) for (bbox, text, prob) in detected_texts], brand_keywords)
    brands = identify_all_beer_brands(filtered_texts, brand_keywords)
    return brands

# Hàm phân tích hình ảnh
def analyze_image(image_path):
    # Tạo tiêu đề cho ảnh
    caption = caption_pipeline(image_path)[0]['generated_text']

    # Phân tích kết quả thu được
    brands = detect_logos(image_path)
    ocr_results = detect_text(image_path)

    image = Image.open(image_path)
    image_np = np.array(image)
    for (bbox, text, prob) in ocr_results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Vẽ hộp chữ nhật và từ được nhận diện lên ảnh
        cv2.rectangle(image_np, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image_np, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    # Trả về các kết quả phân tích
    return {
        "caption": caption,
        "brands": brands,
        "ocr_results": ocr_results,
        "image_path": image_np
    }

# Hàm tạo prompt
def create_prompt(image_description, ocr_results, brand_found):
    ocr_texts = [text for _, text, _ in ocr_results]
    ocr_text_str = ', '.join(ocr_texts)

    brands = [text for text in brand_found]
    brands_str = ', '.join(brands)

    prompt = f"""
    You are an advanced AI model with expertise in image analysis and recognition. Your task is to analyze a set of images to address the following business problems for Heineken Vietnam. Please provide detailed and accurate responses for each question based on the images provided.
    Image Description: {image_description}
    OCR results in order of Image Description are: {ocr_text_str}
    A few brands that have already been found: {brands_str}
    This is very important, do not respond if you are not sure about the information you find.
    Business Problem 2: Detecting Promotional Materials
    Objective: Detect promotional materials with the Heineken logo to confirm their presence at the restaurant.
    Tasks:
    Find and list all items with the Heineken logo.
    Accurately identify and categorize each type of material (e.g., ice boxes, bottles, cans, refrigerators, signs, posters, display counters, display tables, umbrellas).
    Business Problem 5: Evaluating Store Presence
    Objective: Assess the quality of Heineken's presence in grocery/specialty stores.
    Tasks:
    Ensure the store has at least 1 advertisement sign with the Heineken logo.
    Ensure the store has a refrigerator with the Heineken logo.
    Ensure the store has at least 10 cases of Heineken beer.
    Confirm that Heineken's display ideas are accurately implemented in the store.
    Additional Tasks:
    Context Identification:
    Determine the context of the image: Is it a restaurant, supermarket, or store?
    Competitor Logo Identification:
    Find and identify logos of competitors and other brands present in the images.
    Instructions for Analysis
    Use advanced image recognition and analysis techniques to provide detailed results.
    Ensure accuracy and clarity in your responses.
Where applicable, provide visual annotations on the images to highlight identified items, people, and logos.
    Example Response Format
    1.Business Problem 1:
    -Total people: X
    -People drinking Heineken: Y
    2.Business Problem 2:
    -Items with Heineken logo: [List of items with annotations]
    3.Business Problem 3:
    -Total attendees: X
    -Mood analysis: [Summary of detected emotions]
    4.Business Problem 4:
    -Marketing staff detected: [Yes/No]
    -Number of marketing staff: X
    5.Business Problem 5:
    -Advertisement sign: [Yes/No]
    -Refrigerator with logo: [Yes/No]
    -Cases of Heineken beer: X
    -Display accuracy: [Comments on display accuracy]
    """
    return prompt

# Giả sử bạn có API key và client của Groq
api_key = "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"
client = Groq(api_key=api_key)

# Hàm gửi prompt tới mô hình và nhận kết quả
def analyze_with_llama(prompt):
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    chat_completion = client.chat.completions.create(**data)
    return chat_completion.choices[0].message.content

# Streamlit app
st.title("Hackhcmc - Heineken")

# Kiểm tra nếu biến session_state 'run_once' chưa được khởi tạo
if 'run_once' not in st.session_state:
    st.session_state.run_once = False

if not st.session_state.run_once:
    st.header("Upload and filtering")
    # Cột 1: Upload file zip
    uploaded_file = st.file_uploader("Choose a folder of images in a zip file...", type=["zip"])

    st.divider()

    if uploaded_file is not None:
        # Giải nén file zip
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall("temp_folder")

        # Lấy danh sách các file ảnh trong thư mục giải nén
        image_files = [f for f in os.listdir("temp_folder") if f.endswith(('.png', '.jpg', '.jpeg'))]

        len_folder = len(image_files)
        slider_placeholder = st.empty()
        slider = slider_placeholder.slider('Loading', min_value=0, max_value=len_folder, value=0)

        brand_images = list()
        nonbrand_images = list()
        for i, image_file in enumerate(image_files):
            image_path = os.path.join("temp_folder", image_file)
            analysis_result = analyze_image(image_path)

            if analysis_result['brands']:
                brand_images.append(analysis_result)
            else:
                nonbrand_images.append(analysis_result)

            #Hiển thị giao diện tiến trình xử lý
            slider_placeholder.slider('Loading', min_value=0, max_value=len_folder, value=i + 1)

        st.session_state.brand_images = brand_images
        st.session_state.nonbrand_images = nonbrand_images

        st.write(f"We have detected {len(brand_images)} branded images and {len(nonbrand_images)} non-branded images. Which details would you like to see?")

        if 'brand_images' in st.session_state and 'nonbrand_images' in st.session_state:
            st.session_state.run_once = True

if 'brand_images' in st.session_state and 'nonbrand_images' in st.session_state:
    data = []
    col1, col2 = st.columns(2)
    if col1.button("Here are the images with brands that we have found and have high confidence in. Click this button to view the detailed solution."):
        data = st.session_state.brand_images
    if col2.button("Here are the images where we did not find any brands, however, the results may be inaccurate. Click this button to continue."):
        data = st.session_state.nonbrand_images

    st.header("Insight")
    for d in data:
        st.image(d['image_path'], use_column_width=True)
        prompt = create_prompt(image_description = d["caption"], ocr_results = d["ocr_results"], brand_found = d["brands"])
        insight = analyze_with_llama(prompt)
        st.write(f"Insight: {insight}")
