# CodeAlpha_HandwrittenRecognition
# Digit Recognition App

An interactive web application built with **Streamlit** and **TensorFlow/Keras** that recognizes handwritten digits (0–9) in real time. Users can draw a digit on a digital canvas, and the underlying Convolutional Neural Network (CNN) model predicts the digit along with confidence scores for all classes[cite: 3, 4].

---

## Key Features

* **Interactive Canvas:** Draw any single digit directly inside the browser using a custom canvas widget.
* **Real-Time Prediction:** Uses a trained CNN model on the classic MNIST dataset to classify handwritten inputs instantly[cite: 3, 4].
* **Probability Distribution:** Displays confidence scores and probabilities across all digits (0–9)[cite: 3, 4].
* **Clean & Simple UI:** Lightweight and intuitive user interface powered by Streamlit[cite: 3].

---

##  Tech Stack

* **Frontend Framework:** Streamlit
* **Canvas Library:** `streamlit-drawable-canvas`
* **Machine Learning / Deep Learning:** TensorFlow / Keras[cite: 3, 4]
* **Data Processing:** NumPy, Pandas
* **Image Processing:** Pillow (PIL)

---

## 📁 Project Structure

```text
digit-recognition-app/
├── app.py                  # Main Streamlit web app script[cite: 3]
├── mnist_cnn_model.h5      # Pre-trained CNN model[cite: 3]
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation

🚀 How to Run Locally
Clone the repository:

Bash
git clone [https://github.com/Shahd799/CodeAlpha_HandwrittenRecognition.git](https://github.com/Shahd799/CodeAlpha_HandwrittenRecognition.git)
cd your-repository
Install dependencies:

Bash
pip install -r requirements.txt
Run the Streamlit application:

Bash
streamlit run app.py
