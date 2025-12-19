# AgriConnect: An Agricultural Platform

AgriConnect is a comprehensive Flask web application designed to empower farmers and connect them directly with consumers and apartment complexes. It integrates various machine learning models and data-driven features to provide solutions for plant disease detection, crop yield prediction, direct sales, and agricultural education.

## Features

### 1. Plant Disease Detection
*   **Ensemble Modeling:** Utilizes an ensemble of deep learning models (ResNet50, DenseNet121, EfficientNetB3, MobileNetV3) for robust disease identification in various crops.
*   **Crop-Specific Detection:** Dedicated models for lemon, potato, rice, tomato, and cucumber diseases, offering tailored analysis.
*   **Confidence-Based Filtering:** Employs soft and hard voting mechanisms to improve prediction accuracy and filter out irrelevant images.
*   **Disease Information:** Provides detailed descriptions, possible steps for treatment, and image references for identified diseases.

### 2. Crop Yield Prediction
*   **Generalized Yield Prediction:** A "new" model based on RandomForestRegressor for broad crop yield forecasting.
*   **Ensemble Yield Prediction:** An "older" ensemble model combining K-Nearest Neighbors, XGBoost, and K-Means clustering for enhanced prediction accuracy.
*   **Karnataka-Specific Yield Prediction:** A specialized model trained for precise yield predictions in the Karnataka region, considering local factors.

### 3. Direct Sales Platform
*   **Farmer-to-Consumer Matching:** Facilitates direct connections between farmers and consumers based on crop offers and requests.
*   **Farmer-to-Apartment Matching:** Connects farmers with apartment complexes to fulfill bulk crop demands.
*   **Consumer-to-Farmer Matching:** Allows consumers to search for farmers offering specific crops they need.

### 4. Agri Chatbot
*   **AI-Powered Assistant:** Integrates a Gemini-powered chatbot to provide accurate and concise answers on farming practices, crop diseases, weather planning, and rural business.
*   **API Key Management:** Includes robust handling for Gemini API key configuration and error reporting.

### 5. E-Learning Portal
*   **Educational Resources:** Offers a curated collection of e-learning videos, categorized by topics such as by-product making, documentation/registration, startup/marketing guidance, and government schemes/loans. (Content primarily in Kannada and farmer-friendly).

### 6. User Management
*   **Farmer and Consumer Accounts:** Supports separate login and registration for farmers and consumers.
*   **Personalized Dashboards:** Provides dedicated dashboards for both user types.

## Technologies Used

*   **Backend:** Flask (Python)
*   **Machine Learning:** PyTorch, scikit-learn, XGBoost
*   **Data Handling:** Pandas, SQLite3
*   **Frontend:** HTML, CSS, JavaScript (via Jinja2 templates)
*   **AI Chatbot:** Google Gemini API
*   **Other Libraries:** Pillow, Werkzeug

## Installation and Setup

### Prerequisites

*   Python 3.8+
*   pip (Python package installer)

### Steps

1.  **Clone the Repository:**
    git clone https://github.com/your-username/AgriConnect.git
    cd AgriConnect
    2.  **Create a Virtual Environment (Recommended):**
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    3.  **Install Dependencies:**
    
    pip install -r requirements.txt
    4.  **Set Up Gemini API Key (for Chatbot functionality):**
    *   Obtain a Gemini API key from [Google AI Studio](https://ai.google.dev/).
    *   Set the API key as an environment variable:
        # On Windows
        $env:GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY'
        # On macOS/Linux
        export GEMINI_API_KEY='YOUR_GEMINI_API_KEY'
            *   (Optional) You can also specify a different Gemini model:
        # On Windows
        $env:GEMINI_MODEL_NAME = 'gemini-1.5-flash'
        # On macOS/Linux
        export GEMINI_MODEL_NAME='gemini-1.5-flash'
        5.  **Initialize the Database:**
    # This will create the site.db file and tables
    python -c "from app import init_db; init_db()"
    6.  **Run the Application:**
    python app.py
        The application will typically run on `http://127.0.0.1:5000/`. A web browser should automatically open to this URL.


