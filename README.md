# Cartoon-image-generation-

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange)
## 🖼 Application Preview

### Original Image
![Input Image](images/input.png)

### Cartoon Output
![Cartoon Image](images/cartoon.png)

### Web Application
![Web Page](images/webpage.png)

Transform your photos into anime-style cartoons using AnimeGANv2, PyTorch, and Streamlit.

This project allows users to upload an image and instantly convert it into a cartoon-style portrait using deep learning.

🚀 Live Features

✨ Upload your photo
🎭 Convert real images into anime/cartoon style
⚡ Fast image processing using PyTorch
⬇️ Download the generated cartoon image
🖥 Easy-to-use Streamlit web interface

🖼 Application Preview

Cartoon Image
### Web Application
![Web Page](images/webpage.png)
### Original Image
![Input Image](images/input.png)
### Cartoon Output
![Cartoon Image](images/cartoon.png)

🧠 Model Used

This project uses AnimeGANv2, a deep learning model designed for fast anime-style image transformation.

Advantages:

High quality cartoon output

Fast inference

Works well on human faces

🛠 Tech Stack

Python

Streamlit

PyTorch

Torchvision

AnimeGANv2

PIL (Python Imaging Library)


📂 Project Structure
cartoon-yourself-app
│
├── app.py
├── output
│   └── cartoon.png
│
├── images
│   └── demo.png
│
├── requirements.txt
└── README.md
⚙️ Installation
-> Clone Repository
git clone https://github.com/yourusername/cartoon-yourself-app.git
-> Move to Project Folder
cd cartoon-yourself-app
-> Install Dependencies
pip install -r requirements.txt
-> Run the App
streamlit run app.py
📸 How the App Works

->.User uploads an image
->Image is resized and normalized
-> AnimeGANv2 processes the image
-> Cartoon-style output is generated
-> User downloads the cartoon image

🔮 Future Improvements

🎨 Multiple cartoon styles

📹 Cartoon video generation

📱 Mobile-friendly UI

⚡ Faster GPU inference
