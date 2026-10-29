"""Flask web app for the trained Persian Cat-vs-Dog classifier."""
from pathlib import Path
from base64 import b64encode
from io import BytesIO

from flask import Flask, request, render_template_string
from PIL import Image, UnidentifiedImageError

from predict import load_classifier, predict_pil

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "outputs" / "best_model.pth"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB upload limit

HTML = """
<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>تشخیص گربه و سگ با هوش مصنوعی</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 24px; min-height: 100vh;
      font-family: Tahoma, Arial, sans-serif; color: #edf2ff;
      background: radial-gradient(circle at top, #24345b, #101522 65%);
      display: flex; align-items: center; justify-content: center;
    }
    .card {
      width: min(100%, 620px); padding: 28px; border-radius: 24px;
      background: rgba(25, 34, 56, .96); border: 1px solid #354363;
      box-shadow: 0 20px 60px #0005;
    }
    h1 { margin: 0 0 10px; font-size: clamp(24px, 5vw, 34px); }
    .subtitle { color: #bdc9e5; line-height: 1.9; }
    .upload {
      margin-top: 24px; padding: 22px; border: 1px dashed #7186b8;
      border-radius: 16px; background: #151e32;
    }
    input[type=file] { width: 100%; margin-bottom: 18px; color: #dce5fb; }
    button {
      width: 100%; padding: 14px; border: 0; border-radius: 12px;
      background: #8db4ff; color: #101a2e; font-size: 17px;
      font-weight: bold; cursor: pointer;
    }
    button:hover { background: #b0cbff; }
    .result {
      margin-top: 24px; padding: 20px; border-radius: 16px;
      background: #202d48; text-align: center;
    }
    .label { font-size: 30px; font-weight: bold; margin: 10px 0; }
    .confidence { color: #bcd2ff; font-size: 18px; }
    .preview { display: block; max-width: 100%; max-height: 320px;
      object-fit: contain; margin: 18px auto 0; border-radius: 14px; }
    .bar { height: 10px; background: #394866; border-radius: 99px; overflow: hidden; margin: 8px 0 14px; }
    .fill { height: 100%; background: #8db4ff; border-radius: 99px; }
    .error { margin-top: 18px; padding: 12px; border-radius: 12px; background: #572e3b; color: #ffdbe2; }
    .note { color: #9eacc8; font-size: 12px; line-height: 1.8; margin-top: 20px; }
  </style>
</head>
<body>
  <main class="card">
    <h1>🐾 تشخیص گربه یا سگ</h1>
    <div class="subtitle">یک عکس انتخاب کن تا مدل هوش مصنوعی تصویر را بررسی کند.</div>
    <form class="upload" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/jpeg,image/png,image/webp,image/bmp" required>
      <button type="submit">🔎 تشخیص تصویر</button>
    </form>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    {% if result %}
      <section class="result">
        <div>نتیجه پیش‌بینی</div>
        <div class="label">{{ result.label_fa }}</div>
        <div class="confidence">اطمینان مدل: {{ "%.1f"|format(result.confidence * 100) }}٪</div>
        <img class="preview" src="data:image/jpeg;base64,{{ image_data }}" alt="تصویر بارگذاری‌شده">
        <div style="text-align:right;margin-top:20px">گربه: {{ "%.1f"|format(result.cat_probability * 100) }}٪</div>
        <div class="bar"><div class="fill" style="width:{{ result.cat_probability * 100 }}%"></div></div>
        <div style="text-align:right">سگ: {{ "%.1f"|format(result.dog_probability * 100) }}٪</div>
        <div class="bar"><div class="fill" style="width:{{ result.dog_probability * 100 }}%"></div></div>
      </section>
    {% endif %}
    <div class="note">درصد اطمینان تضمین درستی پیش‌بینی نیست. برای نتیجه بهتر از عکس واضحی استفاده کن که حیوان در آن مشخص باشد.</div>
  </main>
</body>
</html>
"""


def get_checkpoint_path() -> Path:
    configured = request.form.get("checkpoint", "").strip() if request.method == "POST" else ""
    return Path(configured).expanduser() if configured else DEFAULT_CHECKPOINT


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    image_data = None
    error = None

    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or not uploaded.filename:
            error = "لطفاً ابتدا یک عکس انتخاب کن."
        else:
            try:
                with Image.open(uploaded.stream) as source:
                    image = source.convert("RGB")
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=90)
                image_data = b64encode(buffer.getvalue()).decode("ascii")

                checkpoint_path = get_checkpoint_path()
                if not checkpoint_path.is_absolute():
                    checkpoint_path = BASE_DIR / checkpoint_path
                model, transform, device = load_classifier(checkpoint_path, device=None)
                result = predict_pil(image, model, transform, device)
            except FileNotFoundError:
                error = f"فایل مدل پیدا نشد: {DEFAULT_CHECKPOINT}. مطمئن شو best_model.pth در پوشه outputs قرار دارد."
            except (UnidentifiedImageError, OSError):
                error = "فایل انتخاب‌شده تصویر معتبر نیست؛ لطفاً JPG یا PNG انتخاب کن."
            except Exception as exc:
                app.logger.exception("Prediction failed")
                error = f"اجرای مدل با خطا مواجه شد: {exc}"

    return render_template_string(HTML, result=result, image_data=image_data, error=error)


@app.errorhandler(413)
def too_large(_error):
    return "حجم عکس بیش از حد مجاز است (حداکثر 12 مگابایت).", 413


if __name__ == "__main__":
    print("Cat-Dog AI is running at http://127.0.0.1:5000")
    print("From another device on the same Wi-Fi, open http://TABLET-IP:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
