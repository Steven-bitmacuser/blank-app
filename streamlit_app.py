import streamlit as st
try:
    import google.generativeai as genai
except ModuleNotFoundError:
    # This might work if the path is slightly misconfigured but the package is installed
    # It tries to access the 'generativeai' folder inside the 'google' namespace.
    try:
        from google import generativeai as genai
    except ModuleNotFoundError as e:
        import streamlit as st
        st.error(f"FATAL ERROR: Failed to import the Google GenAI SDK. Please check your GitHub requirements.txt.")
        st.exception(e)
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
# 🔑 Hardcode your API key here. It will be exposed in your code.
# REPLACE 'YOUR_HARDCODED_GEMINI_API_KEY_HERE' with your actual key.
HARDCODED_API_KEY = "AIzaSyA_U1-epz5hS9wF5mydEMe_Ij0mfzaWsk4"
GEMINI_MODEL = 'gemini-2.5-flash' # Stable and fast model

# --- Helper Functions: OCR and Extraction ---

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
        # Use a temporary directory for safer file handling, though not strictly necessary here
        try:
            file_extension = os.path.splitext(file.name)[1].lower()

            if file_extension == '.pdf':
                st.info(f"Converting PDF: {file.name} to images...")
                # convert_from_bytes requires poppler-utils installed on the system
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
            # pytesseract requires tesseract-ocr installed on the system
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

# --- Sureness Score Function ---

def calculate_sureness_score(df, raw_output):
    """
    Calculates a simple score based on the quality and completeness of the CSV output.
    """
    required_cols = ['Question_Number', 'Marks_Awarded', 'Maximum_Marks', 'Detailed_Feedback']
    max_score = 100
    score = max_score

    # Penalty 1: Missing columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        score -= 40
        
    # Penalty 2: Data type issues (API didn't strictly follow CSV rules)
    try:
        if df['Marks_Awarded'].isnull().any() or df['Maximum_Marks'].isnull().any():
             score -= 20
    except Exception:
         score -= 20

    # Penalty 3: Low number of rows (suggests incomplete marking)
    if df.shape[0] < 3: 
        score -= 15

    # Penalty 4: Output cleanup (if the raw output had non-CSV characters)
    # Check if the output is significantly larger than the resulting CSV
    if len(raw_output) > len(df.to_csv(index=False)) * 1.5:
        score -= 5
        
    return max(0, score) 

# --- Helper Function: API Call ---

def get_marking_from_gemini(content):
    """Calls the Gemini API to get the structured marking data."""
    api_key = HARDCODED_API_KEY 

    if not api_key or api_key == "YOUR_HARDCODED_GEMINI_API_KEY_HERE":
        st.error("Gemini API key is not configured in the script.")
        return None
        
    try:
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
        
        # Adding a 1-second delay to avoid rate limit issues on rapid testing
        time.sleep(1) 
        
        response = model.generate_content(prompt)
        
        # Clean the output (remove markdown fences like ```csv)
        cleaned_response = response.text.strip()
        cleaned_response = re.sub(r'```csv\s*', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'```', '', cleaned_response)
        
        return cleaned_response
    except Exception as e:
        st.error(f"An error occurred while calling the Gemini API: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="IGCSE and A-level Auto-Marker AI", layout="wide")
st.title(" IGCSE/A-level Auto-Marker AI")
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
        # Pre-check for API key before processing large files
        if HARDCODED_API_KEY == "YOUR_HARDCODED_GEMINI_API_KEY_HERE":
            st.error("Please set your Gemini API key in the `HARDCODED_API_KEY` variable at the top of the script.")
            st.stop()
            
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
                    
                    # Ensure mark columns are numeric before calculating score
                    marking_df['Marks_Awarded'] = pd.to_numeric(marking_df['Marks_Awarded'], errors='coerce')
                    marking_df['Maximum_Marks'] = pd.to_numeric(marking_df['Maximum_Marks'], errors='coerce')

                    # Calculate Sureness Score
                    sureness_score = calculate_sureness_score(marking_df.copy(), marking_csv_data)

                    st.header("Marking Report")
                    
                    # Display Metrics
                    st.markdown("---")
                    col1, col2 = st.columns([1, 4])
                    col1.metric(
                        label="**Sureness Score**", 
                        value=f"{sureness_score}/100", 
                        help="An internal metric estimating the quality and consistency of the AI's CSV output."
                    )
                    
                    # Convert to numeric safely for totals (fillna(0) for corrupted rows)
                    marking_df['Marks_Awarded'] = marking_df['Marks_Awarded'].fillna(0).astype(int)
                    marking_df['Maximum_Marks'] = marking_df['Maximum_Marks'].fillna(0).astype(int)

                    total_awarded = marking_df['Marks_Awarded'].sum()
                    total_max = marking_df['Maximum_Marks'].sum()
                    
                    col2.metric(label="**Total Score**", value=f"{total_awarded} / {total_max}")
                    st.markdown("---")
                    
                    # Display the DataFrame with text wrapping enabled for feedback
                    st.dataframe(
                        marking_df,
                        column_config={
                            # Forces text in the 'Detailed_Feedback' column to wrap
                            "Detailed_Feedback": st.column_config.TextColumn(
                                "Detailed Feedback",
                                help="Explanation of marks awarded or not awarded.",
                                width="large" 
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
