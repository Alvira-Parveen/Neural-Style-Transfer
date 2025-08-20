# 🎨 Neural Style Transfer  

Apply the artistic style of one image to another using **Deep Learning** 🧠✨.  
This project implements **Neural Style Transfer (NST)** with **TensorFlow + VGG19**.  

---

## 📌 Features  
- 🖼️ Combine **content image** + **style image** into a stylized output  
- ⚡ Uses **pre-trained VGG19** model  
- 🔧 Easy to run with a single Python script  
- 🐍 Written in Python, runs on macOS (Apple Silicon supported)  

---

## 📂 Project Structure  
Neural Style Transfer/
│── style_transfer.py # Main script
│── content.jpg # content image
│── style.jpg # style image
│── requirements.txt # Dependencies
│── README.md # Project info


---

## 🛠️ Installation  

1. Clone this repo or create a new folder :  
   bash
   git clone <your-repo-link>
   cd Neural\ Style\ Transfer

2. Create & activate virtual environment (recommended):
python3 -m venv nst-env
source nst-env/bin/activate   # Mac/Linux
nst-env\Scripts\activate      # Windows

3. Install dependencies :
pip install -r requirements.txt

---


## ▶️ Usage

1) Place your content image (e.g., content.jpg) and style image (e.g., style.jpg) in the folder.

2) Run the script :
        python style_transfer.py

3) Output: A stylized image will be displayed after training 🎉

---


## 📝 Notes & Tips

- ⚡ **Performance**
  - Use `256×256` or `512×512` images for faster runs. Larger sizes (e.g. 1024×1024) are much slower and memory-hungry.
  - Try `iterations = 200` for a good balance between quality and speed; increase to `300–500` for sharper results.

- 🖼️ **Image selection**
  - Choose a clear content image with a distinct subject (portrait, building, landscape).
  - Prefer style images that have strong, coherent brush strokes or textures (Van Gogh, Monet, etc.). Avoid extremely noisy textures.
  - If style details overwhelm content, reduce `style_weight` or increase `content_weight`.

- 🔧 **Parameter tuning**
  - `style_weight` controls how strongly the style is applied (higher → more stylized).
  - `content_weight` controls content preservation (higher → more like the original photo).
  - Recommended starting values (used in this repo): `style_weight = 1e-2`, `content_weight = 1e4`.

- 💾 **Saving & reproducibility**
  - Save final images (e.g. `stylized_output.jpg`) so you can include before/after examples in the repo.
  - Record the resolution, iteration count, and weights used for each saved result so experiments are reproducible.

---


## 👩‍💻 Author

**Name**  : ALVIRA PARVEEN  
🔗 [LinkedIn](https://www.linkedin.com/in/alvira-parveen-78022536b)  
🌐 [GitHub](https://github.com/Alvira-Parveen)