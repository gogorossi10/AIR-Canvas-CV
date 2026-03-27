🎨 Air Canvas: Gesture-Based Drawing & Arithmetic Recognition

📌 Project Overview

Air Canvas is a computer vision-based application that allows users to draw in the air using **finger movements tracked via a webcam**. The system captures these gestures in real time and displays them on a virtual canvas. Additionally, it integrates Optical Character Recognition (OCR) to interpret and evaluate arithmetic expressions drawn by the user.

---

🚀 Features

* ✋ **Finger-based gesture drawing** (no physical touch required)
* 🎨 **Multi-color selection** (Blue, Green, Red, Yellow)
* 🧹 **Clear Canvas functionality**
* 🔤 **OCR Integration** to recognize and evaluate expressions
* ⚡ **Real-time FPS display**
* 🖥️ **Live camera feed with gesture tracking**

---

🧠 Technologies Used

* Python
* OpenCV (Computer Vision)
* NumPy
* pytesseract (OCR)
* Image Processing Techniques

---

⚙️ Installation & Setup

1️⃣ Clone the Repository


git clone https://github.com/your-username/air-canvas.git
cd air-canvas


---

2️⃣ Install Dependencies

```bash
pip install opencv-python numpy pytesseract
```

---

 3️⃣ Install Tesseract OCR

Download and install from:
https://github.com/tesseract-ocr/tesseract

After installation, add this in your code:


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


---

 ▶️ How to Run


python "Air canvas .py"


---

🎯 How to Use

1. Run the program — your webcam will start.
2. Raise your **index finger** in front of the camera.
3. Move your finger in the air to draw on the screen.

 Toolbar Controls:

| Button                      | Function             |
| --------------------------- | -------------------- |
| CLEAR                       | Clears canvas        |
| BLUE / GREEN / RED / YELLOW | Change drawing color |

4. Draw arithmetic expressions (e.g., `2+3`)

5. The system will:

* Capture the drawing
* Convert it to text using OCR
* Evaluate the expression
* Display the result on screen

6. Press Q to exit

---

📂 Output

* Real-time drawing on screen
* Evaluated arithmetic result displayed on video feed

---

⚠️ Limitations

* Sensitive to lighting conditions
* Finger detection may vary with background
* OCR may misinterpret unclear drawings
* Requires steady hand movement for better accuracy

---

🔮 Future Enhancements

* Save and Undo functionality
* Improved gesture recognition using deep learning
* Multi-hand support
* Mobile/web deployment
* Advanced math recognition

---

📚 Learning Outcomes

* Understanding of computer vision techniques
* Real-time video processing
* Gesture-based human-computer interaction
* Integration of OCR with vision systems
* Debugging and performance optimization

---

 👨‍💻 Author

Gokul Satheesh
B.Tech CSE (AI & ML)
VIT Bhopal University

---

 📌 Conclusion

This project demonstrates how computer vision can be used to build intuitive, touchless interaction systems. By using finger-based gesture recognition and OCR, Air Canvas provides a smart and interactive platform for drawing and solving arithmetic expressions in real time.

---
