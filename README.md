# 🎈 A simple Alevel/IGCSE paper marker AI


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   echo "Creating requirements.txt..."
   cat << EOF > requirements.txt
   streamlit
   google-genai
   pillow
   pytesseract
   pandas
   pdf2image
   EOF

   # Create packages.txt (System dependencies for Tesseract and Poppler)
   echo "Creating packages.txt..."
   cat << EOF > packages.txt
   tesseract-ocr
   poppler-utils
   EOF

   # --- 2. Install Dependencies (For Local/Codespace Testing) ---

   # Update system packages and install Tesseract and Poppler
   echo "Installing system dependencies (Tesseract and Poppler)..."
   sudo apt update -y
   sudo apt install -y tesseract-ocr poppler-utils

   # Install Python packages
   echo "Installing Python dependencies..."
   pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
