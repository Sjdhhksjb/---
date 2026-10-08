import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import io

# إعداد الصفحة
st.set_page_config(page_title="محرر الصور الذكي", layout="wide", page_icon="🎨")
st.title("🎨 تطبيق تعديل الصور بالذكاء الاصطناعي والمعالجة المتقدمة")
st.caption("أداة مجانية لتعديل وتحسين الصور باستخدام الذكاء الاصطناعي والرؤية الحاسوبية")

# شريط الأدوات الجانبي
with st.sidebar:
    st.header("⚙️ خيارات التعديل")
    mode = st.selectbox(
        "اختر وضع المعالجة:",
        [
            "تحسين الجودة والإضاءة (Enhancement)",
            "فلاتر ذكية (Sketch / Cartoon / Edge)",
            "إزالة التشويش وتنعيم الصورة (Denoise)",
            "تحويل النمط الفني (Stylize)"
        ]
    )

# رفع الصورة
uploaded_file = st.file_uploader("قم برفع صورتك هنا (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

def convert_pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def convert_cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("الصورة الأصلية")
        st.image(original_img, use_container_width=True)

    result_img = original_img.copy()

    # معالجة الصور بناءً على الوضع المختار
    if mode == "تحسين الجودة والإضاءة (Enhancement)":
        brightness = st.sidebar.slider("السطوع", 0.5, 2.0, 1.0, 0.1)
        contrast = st.sidebar.slider("التباين", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.sidebar.slider("الحدة (Sharpness)", 0.5, 3.0, 1.0, 0.1)

        enhancer = ImageEnhance.Brightness(result_img)
        result_img = enhancer.enhance(brightness)
        enhancer = ImageEnhance.Contrast(result_img)
        result_img = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Sharpness(result_img)
        result_img = enhancer.enhance(sharpness)

    elif mode == "فلاتر ذكية (Sketch / Cartoon / Edge)":
        filter_type = st.sidebar.radio("نوع الفلتر:", ["رسم بقلم رصاص (Pencil Sketch)", "كرتون (Cartoon Effect)", "تحديد الحواف (Canny Edges)"])
        cv_img = convert_pil_to_cv2(original_img)
        
        if filter_type == "رسم بقلم رصاص (Pencil Sketch)":
            gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            inverted_img = 255 - gray_img
            blurred = cv2.GaussianBlur(inverted_img, (21, 21), 0)
            inverted_blur = 255 - blurred
            sketch = cv2.divide(gray_img, inverted_blur, scale=256.0)
            result_img = Image.fromarray(sketch)
            
        elif filter_type == "كرتون (Cartoon Effect)":
            # تبسيط الألوان مع الحفاظ على الحواف
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(cv_img, 9, 250, 250)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            result_img = convert_cv2_to_pil(cartoon)
            
        elif filter_type == "تحديد الحواف (Canny Edges)":
            edges = cv2.Canny(cv_img, 100, 200)
            result_img = Image.fromarray(edges)

    elif mode == "إزالة التشويش وتنعيم الصورة (Denoise)":
        strength = st.sidebar.slider("شدة التنقية", 3, 20, 10, 1)
        cv_img = convert_pil_to_cv2(original_img)
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, strength, strength, 7, 21)
        result_img = convert_cv2_to_pil(denoised)

    elif mode == "تحويل النمط الفني (Stylize)":
        cv_img = convert_pil_to_cv2(original_img)
        stylized = cv2.stylization(cv_img, sigma_s=60, sigma_r=0.07)
        result_img = convert_cv2_to_pil(stylized)

    # عرض النتيجة
    with col2:
        st.subheader("النتيجة المعدلة")
        st.image(result_img, use_container_width=True)

        # زر تحميل النتيجة
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="💾 تحميل الصورة المعدلة",
            data=byte_im,
            file_name="edited_image.png",
            mime="image/png"
        )
