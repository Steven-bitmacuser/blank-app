import streamlit as st
# The standard import works correctly when the API key is passed via secrets
# because it forces the correct environment configuration.
import google.generativeai as genai 

# Fallback in case the new import fails for some reason
try:
    import google.generativeai as genai
except ImportError:
    try:
        import google_genai as genai
    except ImportError:
        st.error("FATAL ERROR: The Google GenAI SDK cannot be imported.")
        st.info("Please ensure 'google-genai' is in your requirements.txt.")
        st.stop()


from PIL import Image
import pytesseract
import io
import pandas as pd
from pdf2image import convert_from_bytes
import os
from io import StringIO
import re # For cleaning up API output
import time # For time-based uniqueness (optional, but good practice)

# --- Configuration ---
# 🔑 SECURITY FIX: The API key is now loaded securely from Streamlit Secrets.
GEMINI_API_KEY_NAME = "GEMINI_API_KEY" # Key name in secrets.toml
GEMINI_MODEL = 'gemini-2.5-flash' # Stable and fast model

# --- Helper Functions: OCR and Extraction (No Changes) ---

def process_and_extract_text(uploaded_files, file_type="Paper"):
    """
    Handles both image and PDF files, converting PDFs to images first, 
    and then performing OCR on all pages/images. Requires system installations 
    of tesseract-ocr and poppler-utils to work.
    """
    st.info(f"Processing {file_type} files...")
    full_text = ""
    images_to_process = []
    
    for file in uploaded_files:
        try:
            file_extension = os.path.splitext(file.name)[1].lower()

            if file_extension == '.pdf':
                st.info(f"Converting PDF: {file.name} to images...")
                pdf_images = convert_from_bytes(file.read())
                images_to_process.extend(pdf_images)
                st.success(f"Converted {len(pdf_images)} pages from {file.name}.")
            elif file_extension in ['.png', '.jpg', '.jpeg']:
                images_to_process.append(Image.open(file))
            else:
                st.warning(f"Skipping unsupported file type: {file.name}")
                continue

        except Exception as e:
            st.error(f"Error preparing file {file.name}. Ensure poppler-utils is installed for PDF: {e}")
            return None

    if not images_to_process:
        st.error(f"No usable content found in {file_type} files.")
        return None

    # Perform Tesseract OCR on all collected images
    for i, image in enumerate(images_to_process):
        try:
            text = pytesseract.image_to_string(image)
            full_text += f"[Page {i+1} Text]\n" + text + "\n\n--- End of Page ---\n\n"
        except Exception as e:
            st.error(f"Error performing OCR on image {i+1}. Ensure tesseract-ocr is installed: {e}")
            return None
            
    st.success(f"Successfully extracted text from {len(images_to_process)} pages of {file_type}.")
    return full_text

def structure_content(paper_text, scheme_text):
    """Packages the extracted text into a dictionary for the API call."""
    if paper_text and scheme_text:
        return {
            "student_paper_text": paper_text,
            "mark_scheme_text": scheme_text
        }
    return None

def calculate_sureness_score(df, raw_output):
    """Calculates a simple score based on the quality and completeness of the CSV output."""
    required_cols = ['Question_Number', 'Marks_Awarded', 'Maximum_Marks', 'Detailed_Feedback']
    max_score = 100
    score = max_score

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        score -= 40
        
    try:
        if 'Marks_Awarded' in df.columns and df['Marks_Awarded'].isnull().any():
             score -= 20
        if 'Maximum_Marks' in df.columns and df['Maximum_Marks'].isnull().any():
             score -= 20
    except Exception:
        score -= 20 

    if df.shape[0] < 3: 
        score -= 15

    if len(raw_output) > len(df.to_csv(index=False)) * 1.5:
        score -= 5
        
    return max(0, score) 

# --- Helper Function: API Call (BIG CHANGE HERE) ---

def get_marking_from_gemini(content):
    """Calls the Gemini API to get the structured marking data, reading key from st.secrets."""
    
    # --- GET KEY SECURELY ---
    try:
        api_key = st.secrets[GEMINI_API_KEY_NAME]
    except KeyError:
        st.error(f"🔑 Gemini API key not found in Streamlit Secrets. Please add '{GEMINI_API_KEY_NAME}' to your secrets file.")
        return None
    
    if not api_key:
        st.error(f"🔑 Gemini API key is empty in Streamlit Secrets.")
        return None
        
    try:
        # Configure the client using the securely retrieved key
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""
        **Role:** You are an expert IGCSE or A-level examiner. Your task is to mark a student's exam paper based on the provided mark scheme.

        **Instructions:**
        1.  Carefully read the student's answers from the "STUDENT'S PAPER" section.
        2.  Compare each answer against the criteria in the "MARK SCHEME" section.
        3.  Award marks strictly according to the mark scheme.
        4.  For each question, provide a detailed breakdown.
        5.  Your final output MUST be a table in CSV format with NO extra text or conversational filler.
        6.  The table MUST have the following columns: 'Question_Number', 'Marks_Awarded', 'Maximum_Marks', 'Detailed_Feedback'.
        7.  The 'Detailed_Feedback' column must explain WHY marks were awarded or not awarded.

        **[STUDENT'S PAPER]**
        {content['student_paper_text']}

        **[MARK SCHEME]**
        {content['mark_scheme_text']}

        **Output (CSV Format):**
        """
        st.info(f"🤖 Calling the Gemini API ({GEMINI_MODEL}) for marking... This might take a moment.")
        
        time.sleep(1) 
        
        response = model.generate_content(prompt)
        
        cleaned_response = response.text.strip()
        cleaned_response = re.sub(r'```csv\s*', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'```', '', cleaned_response)
        
        return cleaned_response
    except Exception as e:
        st.error(f"An error occurred while calling the Gemini API: {e}")
        return None

# --- Streamlit UI (No Changes) ---
st.set_page_config(page_title="IGCSE and A-level Auto-Marker AI", layout="wide")
st.title("👨‍🏫 IGCSE/A-level Auto-Marker AI")
st.write("Upload an exam paper and its mark scheme (Images or PDF) to get an automated, detailed marking report.")
st.write("BEWARE!!! The images/PDFs needed to be horizontally aligned for better OCR results.")
st.sidebar.header("Upload Files")
file_types = ["png", "jpg", "jpeg", "pdf"] 
paper_files = st.sidebar.file_uploader("Upload Student's Paper (Images/PDF)", type=file_types, accept_multiple_files=True)
scheme_files = st.sidebar.file_uploader("Upload Mark Scheme (Images/PDF)", type=file_types, accept_multiple_files=True)

if st.sidebar.button("✨ Mark Paper"):
    if not paper_files or not scheme_files:
        st.warning("Please upload both the paper and the mark scheme files.")
    else:
        # NOTE: Removed the hardcoded key check here, it's now done in get_marking_from_gemini()
        
        with st.spinner('Reading and processing files (OCR)...'):
            student_paper_text = process_and_extract_text(paper_files, file_type="Paper")
            mark_scheme_text = process_and_extract_text(scheme_files, file_type="Mark Scheme")
        
        if student_paper_text and mark_scheme_text:
            structured_content = structure_content(student_paper_text, mark_scheme_text)
            marking_csv_data = get_marking_from_gemini(structured_content) 
            
            if marking_csv_data:
                st.success("🎉 Marking Complete!")
                try:
                    csv_io = StringIO(marking_csv_data)
                    marking_df = pd.read_csv(csv_io)
                    
                    marking_df['Marks_Awarded'] = pd.to_numeric(marking_df['Marks_Awarded'], errors='coerce')
                    marking_df['Maximum_Marks'] = pd.to_numeric(marking_df['Maximum_Marks'], errors='coerce')

                    sureness_score = calculate_sureness_score(marking_df.copy(), marking_csv_data)

                    st.header("Marking Report")
                    
                    st.markdown("---")
                    col1, col2 = st.columns([1, 4])
                    col1.metric(
                        label="**Sureness Score**", 
                        value=f"{sureness_score}/100", 
                        help="An internal metric estimating the quality and consistency of the AI's CSV output."
                    )
                    
                    marking_df['Marks_Awarded'] = marking_df['Marks_Awarded'].fillna(0).astype(int)
                    marking_df['Maximum_Marks'] = marking_df['Maximum_Marks'].fillna(0).astype(int)

                    total_awarded = marking_df['Marks_Awarded'].sum()
                    total_max = marking_df['Maximum_Marks'].sum()
                    
                    col2.metric(label="**Total Score**", value=f"{total_awarded} / {total_max}")
                    st.markdown("---")
                    
                    st.dataframe(
                        marking_df,
                        column_config={
                            "Detailed_Feedback": st.column_config.TextColumn(
                                "Detailed Feedback", help="Explanation of marks awarded or not awarded.", width="large" 
                            ),
                            "Question_Number": st.column_config.TextColumn(width="small"),
                            "Marks_Awarded": st.column_config.TextColumn(width="small"),
                            "Maximum_Marks": st.column_config.TextColumn(width="small"),
                        },
                        height=500, 
                        use_container_width=True
                    )

                except pd.errors.ParserError:
                    st.error("Could not parse the marking data into a table. The AI output was not clean CSV.")
                    st.subheader("Raw Gemini Output for Debugging:")
                    st.text_area("Raw Output", marking_csv_data, height=300)
                except Exception as e:
                    st.error(f"An unexpected error occurred during report generation: {e}")
