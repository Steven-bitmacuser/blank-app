# streamlit_app.py
import streamlit as st
import os
import io
import re
import time
import base64
import shutil
from io import StringIO
from PIL import Image
import pandas as pd

# ==========  IMPORT GOOGLE GENERATIVE AI (official package) ==========
try:
    import google.generativeai as genai
except Exception as e:
    # Friendly error: stops app with clear instruction if import fails
    st.set_page_config(page_title="IGCSE/A-level Auto-Marker AI", layout="wide")
    st.title("IGCSE/A-level Auto-Marker AI")
    st.error("FATAL ERROR: The Google Generative AI SDK could not be imported.")
    st.info("Make sure your requirements.txt contains: google-generativeai>=0.5.4")
    st.exception(e)
    st.stop()

# ========== STREAMLIT PAGE CONFIG ==========
st.set_page_config(page_title="IGCSE/A-level Auto-Marker AI", layout="wide", page_icon="🤌")
st.title("IGCSE/A-level Auto-Marker AI (V4) - Fixed CSV Logic")
st.write("Upload student's paper and mark scheme (images or PDF). This version includes improved CSV validation, normalization, and a more honest sureness score.")

# ========= Sidebar: Uploads and Options ==========
st.sidebar.header("1) Upload Files")
file_types = ["png", "jpg", "jpeg", "pdf"]
paper_files = st.sidebar.file_uploader("Student's Paper (Images / PDF)", type=file_types, accept_multiple_files=True)
scheme_files = st.sidebar.file_uploader("Mark Scheme (Images / PDF)", type=file_types, accept_multiple_files=True)

st.sidebar.header("2) Marking options")
expected_full_marks = st.sidebar.number_input("Expected Full Marks for paper (enter exact integer)", min_value=1, value=75, step=1, help="Tell the app the paper's maximum total marks (e.g., 75). This helps validation.")
auto_normalize = st.sidebar.checkbox("Auto-normalize totals if AI output's total differs", value=True, help="If the AI's 'Maximum_Marks' total doesn't equal Expected Full Marks, scale results to match.")
debug_mode = st.sidebar.checkbox("Debug mode (show raw AI output & logs)", value=False)

st.sidebar.header("3) System check (OCR prerequisites)")
if shutil.which("tesseract"):
    st.sidebar.success("✅ Tesseract installed")
else:
    st.sidebar.error("❌ Tesseract not found (system package missing)")

if shutil.which("pdftoppm"):
    st.sidebar.success("✅ Poppler (pdftoppm) installed")
else:
    st.sidebar.error("❌ Poppler not found (system package missing)")

st.sidebar.markdown("---")
st.sidebar.caption("If Tesseract/Poppler are not installed on Streamlit Cloud, please ensure packages.txt at repo root contains:\n```\ntesseract-ocr\npoppler-utils\n```")

# ========== Helper: OCR & PDF to images ==========
# These imports rely on system packages (tesseract-ocr and poppler-utils)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
except ImportError as e:
    st.warning(f"Some OCR libraries failed to import: {e}. Check system prerequisites.")
    # Define dummy functions to prevent crash if libraries are missing but app is run
    def convert_from_bytes(*args, **kwargs): raise ImportError("pdf2image not found.")
    class DummyTesseract:
        @staticmethod
        def image_to_string(*args, **kwargs): raise ImportError("pytesseract not found.")
    pytesseract = DummyTesseract()


def process_and_extract_text(uploaded_files, file_type="Paper"):
    """
    Convert PDFs -> images (pdf2image) and run pytesseract on images.
    Returns concatenated text with page delimiters.
    """
    st.info(f"Processing {file_type} files...")
    images_to_process = []
    full_text = ""

    for file in uploaded_files:
        try:
            filename = getattr(file, "name", "uploaded_file")
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".pdf":
                st.info(f"Converting PDF to images: {filename}")
                # convert_from_bytes relies on poppler (pdftoppm) on system
                pdf_bytes = file.read()
                pdf_images = convert_from_bytes(pdf_bytes)
                images_to_process.extend(pdf_images)
                st.info(f"Converted {len(pdf_images)} pages from {filename}")
            elif ext in [".png", ".jpg", ".jpeg"]:
                # file is an UploadedFile (BytesIO-like). PIL can open it.
                img = Image.open(file)
                images_to_process.append(img)
            else:
                st.warning(f"Unsupported file type: {filename}. Skipped.")
        except ImportError:
            st.error(f"Error: Required OCR/PDF dependency is missing for {filename}. Check system prerequisites (tesseract/poppler).")
            return None
        except Exception as e:
            st.error(f"Error preparing file {filename}: {e}")
            if debug_mode:
                st.exception(e)
            return None

    if not images_to_process:
        st.error(f"No images/pages found in {file_type} files.")
        return None

    for i, img in enumerate(images_to_process):
        try:
            text = pytesseract.image_to_string(img)
            full_text += f"[Page {i+1}]\n" + text + "\n\n"
        except ImportError:
            st.error(f"Error: Tesseract is not configured correctly on the system.")
            return None
        except Exception as e:
            st.error(f"Error performing OCR on page {i+1}: {e}")
            if debug_mode:
                st.exception(e)
            return None

    st.success(f"Extracted text from {len(images_to_process)} pages of {file_type}.")
    return full_text

# ========== Helper: create a robust prompt for Gemini (FIXED CSV QUOTING) ==========
GEMINI_MODEL = "gemini-2.5-flash"

def build_marking_prompt(student_text, scheme_text, expected_full_marks):
    """
    Build a strict prompt that forces CSV output and requires the sum of Maximum_Marks to equal expected_full_marks.
    FIX: Added explicit instruction to quote the Detailed_Feedback column to prevent commas from breaking CSV.
    """
    prompt = f"""
You are a professional IGCSE/A-level examiner. Your job is to mark the student's paper strictly using the following mark scheme.

REQUIREMENTS (MUST FOLLOW EXACTLY):
1. Output pure CSV only. NO extra text, no explanations, no markdown fences.
2. Columns MUST be: Question_Number,Marks_Awarded,Maximum_Marks,Detailed_Feedback
3. Question_Number examples: 1, 2, 3a, 3b, 4(i), etc. Use the same labels that appear in the student's paper.
4. Marks_Awarded and Maximum_Marks MUST be integers. Do not use ranges, text, or fractions.
5. The sum of the Maximum_Marks column MUST equal exactly {expected_full_marks}.
6. Do NOT include any "Total" or "Summary" rows. Only question rows.
7. Detailed_Feedback must be 1-2 concise sentences explaining why marks were awarded or not. **CRITICAL: This field MUST be enclosed in double quotation marks (")** to ensure the CSV structure is not corrupted by commas within the feedback text.

Now produce the CSV table only.

[STUDENT'S PAPER]
{student_text}

[MARK SCHEME]
{scheme_text}

Produce the CSV now:
"""
    return prompt

# ========== Helper: call Gemini and try to get a cleaned CSV string ==========
def call_gemini_for_csv(content_dict, expected_full_marks, max_retries=2):
    """
    Calls Gemini with a strict CSV prompt, attempts to clean the response, and retries
    with additional instructions if CSV is invalid or totals mismatch.
    Returns: cleaned_response_text (string) or None, plus list of debug logs.
    """
    logs = []
    api_key_name = "GEMINI_API_KEY"
    try:
        api_key = st.secrets[api_key_name]
    except KeyError:
        st.error(f"🔑 Gemini API key not found in Streamlit Secrets. Add '{api_key_name}' to secrets.")
        return None, ["Missing API key"]

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    base_prompt = build_marking_prompt(content_dict["student_paper_text"], content_dict["mark_scheme_text"], expected_full_marks)

    attempt = 0
    last_response_text = None

    while attempt <= max_retries:
        attempt += 1
        logs.append(f"Attempt {attempt}: sending prompt (length {len(base_prompt)}).")
        try:
            # Primary call
            response = model.generate_content(base_prompt)
            raw = response.text if hasattr(response, "text") else str(response)
            logs.append(f"Raw response length: {len(raw)}")
        except Exception as e:
            logs.append(f"Call failed on attempt {attempt}: {e}")
            if debug_mode:
                st.exception(e)
            return None, logs

        # Clean: remove markdown fences if present
        cleaned = re.sub(r'```(?:csv)?', '', raw, flags=re.IGNORECASE).strip()
        last_response_text = cleaned

        # Quick heuristic: does it look like CSV? (has commas + header)
        if "Question_Number" in cleaned and "Marks_Awarded" in cleaned and "Maximum_Marks" in cleaned:
            # return cleaned if passes basic check
            logs.append("Basic CSV header found in AI output.")
            return cleaned, logs
        else:
            logs.append("CSV header not found in AI output.")
            # If not found and still retries left, craft a repair prompt
            if attempt <= max_retries:
                repair_prompt = f"""
The table you previously returned is not valid CSV or misses the required columns.
Remember: output ONLY CSV with EXACT columns:
Question_Number,Marks_Awarded,Maximum_Marks,Detailed_Feedback
Ensure the sum of Maximum_Marks equals exactly {expected_full_marks}.
**CRITICAL: The Detailed_Feedback field MUST be enclosed in double quotes (")**.
Return ONLY the CSV, nothing else.
Here is your previous output (for reference):

{cleaned}
"""
                base_prompt = repair_prompt  # next loop will send repair prompt
                logs.append("Prepared repair prompt for next attempt.")
                time.sleep(0.6)  # small delay before retry
                continue
            else:
                logs.append("Max retries reached and no valid CSV produced.")
                return cleaned, logs

    return last_response_text, logs

# ========== Helper: parse CSV text into DataFrame and validate (IMPROVED VALIDATION) ==========
def parse_marking_csv_text(csv_text):
    """
    Tries to parse csv_text into pandas DataFrame.
    Returns (df, parse_error_message or None)
    """
    try:
        csv_io = StringIO(csv_text)
        # Using the 'python' engine for better handling of quoted fields, which is the fix implemented in the prompt
        df = pd.read_csv(csv_io, sep=',', header=0, skipinitialspace=True, engine='python')
        
        # Standardize column names by stripping whitespace
        df.columns = [c.strip() for c in df.columns]
        expected_cols = ['Question_Number', 'Marks_Awarded', 'Maximum_Marks', 'Detailed_Feedback']
        
        # Check for correct number of columns (exactly 4)
        if len(df.columns) != 4 or not all(col in df.columns for col in expected_cols):
            missing = [c for c in expected_cols if c not in df.columns]
            return None, f"Missing or incorrect columns. Expected {expected_cols}, found {list(df.columns)}. Missing: {missing}"

        # Ensure numeric where needed
        df['Marks_Awarded'] = pd.to_numeric(df['Marks_Awarded'], errors='coerce')
        df['Maximum_Marks'] = pd.to_numeric(df['Maximum_Marks'], errors='coerce')
        return df, None
    except Exception as e:
        return None, f"CSV parse error: {e}"

# ========== Improved sureness score ==========
def calculate_sureness_score(df, raw_output, expected_full_marks):
    """
    Returns a 0-100 sureness score. More honest and penalizes:
    - missing columns
    - NaNs in marks
    - total max mismatch
    - very few rows
    - extraneous characters in raw output
    """
    max_score = 100
    score = max_score
    req_cols = ['Question_Number', 'Marks_Awarded', 'Maximum_Marks', 'Detailed_Feedback']

    # Penalize missing columns (already handled in parse, but kept for robustness)
    missing_cols = [c for c in req_cols if c not in df.columns]
    if missing_cols:
        score -= 40

    # Penalize NaNs in mark columns
    try:
        # Check if any original NaNs were present before the final fillna(0)
        if df['Marks_Awarded'].isnull().any():
            score -= 15
        if df['Maximum_Marks'].isnull().any():
            score -= 15
    except Exception:
        score -= 20

    # Penalize very few rows
    if df.shape[0] < 3:
        score -= 10

    # Penalize mismatch in full marks
    total_max = df['Maximum_Marks'].sum()
    if total_max != expected_full_marks:
        # larger penalty when difference is larger
        diff = abs(total_max - expected_full_marks)
        if diff <= 2:
            score -= 5
        elif diff <= 5:
            score -= 12
        else:
            score -= 25

    # Penalize raw text length vs CSV length (excess filler)
    try:
        # Use a copy to avoid converting non-numeric data to float on the main DF
        csv_len = len(df.to_csv(index=False))
        # Raw output shouldn't be much longer than the CSV text itself
        if len(raw_output) > csv_len * 1.6:
            score -= 5
    except Exception:
        score -= 5

    return max(0, score)

# ========== Streamlit UI: main action ==========
if st.button("✨ Mark Paper"):
    if not paper_files or not scheme_files:
        st.warning("Please upload both the student's paper and the mark scheme.")
    else:
        # Use a single spinner for the long OCR phase
        with st.spinner("Performing OCR on uploaded files... (This may take a moment for PDFs)"):
            student_paper_text = process_and_extract_text(paper_files, file_type="Paper")
            mark_scheme_text = process_and_extract_text(scheme_files, file_type="Mark Scheme")

        if not student_paper_text or not mark_scheme_text:
            st.error("OCR failed or returned no text. See sidebar system checks or enable debug mode for more info.")
        else:
            structured = {
                "student_paper_text": student_paper_text,
                "mark_scheme_text": mark_scheme_text
            }
            st.info("Calling the Gemini model to produce the CSV marking table (may take 10-20 seconds)...")

            raw_csv_text, call_logs = call_gemini_for_csv(structured, expected_full_marks, max_retries=2)

            if debug_mode:
                st.subheader("Call Logs")
                for log in call_logs:
                    st.text(log)

            if not raw_csv_text:
                st.error("The AI did not return a usable CSV. See logs above.")
            else:
                # Show raw output (collapsible)
                with st.expander("Raw AI Output (click to expand)", expanded=debug_mode):
                    st.code(raw_csv_text[:5000])  # show up to 5000 chars

                # --- Parsing and Validation ---
                df, parse_err = parse_marking_csv_text(raw_csv_text)
                if parse_err:
                    st.error(f"AI output couldn't be parsed as CSV: {parse_err}. **This usually means the AI failed to correctly quote the Detailed_Feedback field.**")
                    if debug_mode:
                        st.subheader("Raw output for debugging")
                        st.text_area("Raw Output", raw_csv_text, height=300)
                    st.stop()

                # Coerce numeric and fill NaN marks with 0 to avoid errors
                # Store the original data before potential normalization for sureness score calculation
                df['Marks_Awarded_Raw'] = df['Marks_Awarded'].copy()
                df['Maximum_Marks_Raw'] = df['Maximum_Marks'].copy()
                
                df['Marks_Awarded'] = df['Marks_Awarded'].fillna(0).astype(float)
                df['Maximum_Marks'] = df['Maximum_Marks'].fillna(0).astype(float)

                # Totals
                total_awarded = df['Marks_Awarded'].sum()
                total_max = df['Maximum_Marks'].sum()

                # If totals mismatch expected full marks, offer options
                if total_max != expected_full_marks:
                    st.warning(f"AI reported total maximum marks = {int(total_max)} but expected = {int(expected_full_marks)}. Applying normalization...")
                    if auto_normalize:
                        # scale factor for Maximum_Marks and Marks_Awarded proportionally
                        if total_max > 0:
                            factor = expected_full_marks / total_max
                            # Add original columns if normalization is done
                            if 'Maximum_Marks_Original' not in df.columns:
                                df['Maximum_Marks_Original'] = df['Maximum_Marks_Raw']
                                df['Marks_Awarded_Original'] = df['Marks_Awarded_Raw']
                            
                            df['Maximum_Marks'] = (df['Maximum_Marks'] * factor).round().astype(int)
                            df['Marks_Awarded'] = (df['Marks_Awarded'] * factor).round().astype(int)
                            total_awarded = int(df['Marks_Awarded'].sum())
                            total_max = int(df['Maximum_Marks'].sum())
                            st.success(f"Auto-normalized marks to match expected full marks ({expected_full_marks}).")
                        else:
                            st.error("AI reported total maximum marks equal to 0; cannot normalize.")
                    else:
                        st.info("Normalization skipped. Enable 'Auto-normalize' in sidebar for automatic scaling.")
                
                # Final numeric coercion (safe)
                df['Marks_Awarded'] = pd.to_numeric(df['Marks_Awarded'], errors='coerce').fillna(0).astype(int)
                df['Maximum_Marks'] = pd.to_numeric(df['Maximum_Marks'], errors='coerce').fillna(0).astype(int)

                # Calculate sureness score (use raw/original columns for this)
                sureness = calculate_sureness_score(df.copy(), raw_csv_text, expected_full_marks)

                # Prepare final DataFrame for display (remove temporary raw columns)
                cols_to_drop = [col for col in ['Marks_Awarded_Raw', 'Maximum_Marks_Raw'] if col in df.columns]
                df = df.drop(columns=cols_to_drop, errors='ignore')

                # Display report
                st.header("Marking Report")
                col1, col2 = st.columns([1, 3])
                col1.metric("Sureness Score", f"{sureness}/100")
                col2.metric("Total Score", f"{int(df['Marks_Awarded'].sum())} / {int(df['Maximum_Marks'].sum())}")

                st.markdown("---")
                # Show table with wrap for feedback
                try:
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=500
                    )
                except Exception:
                    st.write(df)

                # Provide raw outputs & download options
                st.markdown("---")
                st.subheader("Download Results")
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                b64 = base64.b64encode(csv_bytes).decode()
                st.markdown(f"[Download CSV](data:file/csv;base64,{b64})")

                if debug_mode:
                    st.subheader("Debug information")
                    st.write("Call logs:")
                    for l in call_logs:
                        st.text(l)
                    st.write("Raw AI output preview:")
                    st.code(raw_csv_text[:4000])

                st.success("Marking complete.")

# ========== Helpful footer ==========
st.markdown("---")
st.caption("Notes: This app uses OCR (pytesseract + pdf2image). For reliable OCR on Streamlit Cloud, ensure packages.txt at repo root includes tesseract-ocr and poppler-utils. If system dependencies are missing, results may fail. The app attempts to make the AI output reliable via stricter prompts, retries, and optional normalization.")
