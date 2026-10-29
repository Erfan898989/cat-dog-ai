# پروژه تشخیص گربه و سگ با یادگیری عمیق 🐱🐶

یک پروژه‌ی کامل پایتون برای طبقه‌بندی تصویر با **PyTorch**؛ شامل CNN اختصاصی، MobileNetV3، ResNet50، افزایش داده، تقسیم train/validation/test، توقف زودهنگام، ذخیره‌ی بهترین مدل، ارزیابی و رابط وب.

## ویژگی‌ها

- چهار مدل قابل انتخاب: `custom_cnn`، `mobilenet_v3_small`، `mobilenet_v3_large` و `resnet50`
- انتقال یادگیری (Transfer Learning) با وزن‌های از پیش‌آموزش‌دیده‌ی ImageNet برای MobileNet و ResNet
- تشخیص خودکار ساختارهای رایج دیتاست: پوشه‌های `Cat/Dog`، `cats/dogs` یا فایل‌های `cat.0.jpg` و `dog.0.jpg`
- بررسی و رد فایل‌های تصویری خراب قبل از آموزش
- تقسیم تصادفی و طبقه‌بندی‌شده به ۸۰٪ آموزش، ۱۰٪ اعتبارسنجی و ۱۰٪ آزمون (قابل تنظیم)
- Mixed Precision روی CUDA، کاهش خودکار نرخ یادگیری و Early Stopping
- خروجی‌های ارزیابی: دقت، precision/recall/F1، ماتریس درهم‌ریختگی و نمودار روند آموزش
- پیش‌بینی از طریق ترمینال (`predict.py`) یا رابط وب فارسی (Streamlit)

> نکته: دقت نهایی به کیفیت داده، سخت‌افزار و تنظیمات آموزش بستگی دارد؛ این پروژه عدد دقت ساختگی یا تضمینی اعلام نمی‌کند.

## ۱) نصب

پیشنهاد: Python 3.11 یا 3.12.

در PowerShell ویندوز، داخل پوشه‌ی پروژه اجرا کن:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

اگر دستور `py -3.11` روی سیستم کار نکرد، از نسخه‌ی پایتونی که نصب داری با `python -m venv .venv` استفاده کن.

**برای کارت گرافیک NVIDIA:** بسته‌ی سازگار CUDA را از راهنمای رسمی نصب PyTorch انتخاب کن. اگر آموزش روی CPU انجام شود، کد همچنان اجرا می‌شود اما معمولاً کندتر است.

## ۲) دریافت دیتاست

دیتاست رسمی Kaggle Dogs vs. Cats در صفحه‌ی زیر قرار دارد:

https://www.kaggle.com/competitions/dogs-vs-cats/data

فایل `train.zip` را دانلود کن. سپس اگر فایل در پوشه‌ی `data` قرار دارد:

```powershell
python prepare_data.py --zip data\train.zip --output data\images
```

این اسکریپت فایل ZIP را در `data/images` استخراج می‌کند. نیازی نیست عکس‌ها را دستی به پوشه‌های جداگانه منتقل کنی؛ اسم‌هایی مثل `cat.0.jpg` و `dog.0.jpg` به‌صورت خودکار شناسایی می‌شوند.

اگر دیتاست از قبل به شکل `PetImages\Cat` و `PetImages\Dog` استخراج شده، مرحله‌ی بالا را رد کن و مسیر پوشه‌ی والد را به `--data` بده.

## ۳) آموزش مدل پیشنهادی

برای شروع، MobileNetV3-Large معمولاً تعادل خوبی بین سرعت و کیفیت دارد:

```powershell
python train.py --data data\images --model mobilenet_v3_large --epochs 12 --batch-size 32
```

اگر حافظه‌ی کارت گرافیک محدود است یا سیستم کند می‌شود:

```powershell
python train.py --data data\images --model mobilenet_v3_small --epochs 10 --batch-size 16
```

برای مقایسه با ResNet50:

```powershell
python train.py --data data\images --model resnet50 --epochs 10 --batch-size 16
```

برای تمرین با CNN اختصاصی از صفر:

```powershell
python train.py --data data\images --model custom_cnn --epochs 20 --batch-size 32 --no-pretrained
```

**آزمایش سریع قبل از آموزش کامل:**

```powershell
python train.py --data data\images --model mobilenet_v3_small --epochs 2 --batch-size 16 --max-per-class 300
```

`--max-per-class 300` فقط ۳۰۰ تصویر از هر کلاس را برای آزمایش می‌گیرد؛ برای نتیجه‌ی جدی، آن را حذف کن.

### وزن‌های ازپیش‌آموزش‌دیده و اینترنت

در اجرای اول MobileNet/ResNet ممکن است PyTorch وزن‌های ImageNet را دانلود کند. اگر اینترنت نداری یا می‌خواهی بدون آن اجرا کنی، گزینه‌ی `--no-pretrained` را اضافه کن؛ آموزش از صفر معمولاً به داده و زمان بیشتری نیاز دارد.

## ۴) پیش‌بینی یک عکس

پس از پایان آموزش، بهترین مدل در `outputs/best_model.pth` ذخیره می‌شود:

```powershell
python predict.py "C:\Path\To\your_photo.jpg"
```

یا مسیر مدل را مشخص کن:

```powershell
python predict.py "C:\Path\To\your_photo.jpg" --checkpoint outputs\best_model.pth
```

## ۵) رابط وب فارسی

```powershell
streamlit run app.py
```

آدرس محلی‌ای که Streamlit در ترمینال نشان می‌دهد را در مرورگر باز کن، عکس انتخاب کن و نتیجه را ببین.

## ۶) خروجی‌ها

بعد از آموزش، در پوشه‌ی `outputs` این فایل‌ها را می‌بینی:

- `best_model.pth` — وزن‌های بهترین مدل بر اساس خطای اعتبارسنجی
- `test_metrics.json` — نتیجه‌ی آزمون و گزارش معیارها
- `history.json` — تاریخچه‌ی loss، accuracy و نرخ یادگیری
- `training_curves.png` — نمودار آموزش و اعتبارسنجی
- `confusion_matrix.png` — ماتریس درهم‌ریختگی روی مجموعه‌ی آزمون

## ساختار کلی پروژه

```text
cat_dog_ai/
├── train.py
├── predict.py
├── app.py
├── prepare_data.py
├── requirements.txt
├── README_FA.md
├── src/
│   ├── data.py
│   ├── model.py
│   └── utils.py
├── data/
└── outputs/
```

## نکته‌های مهم برای نتیجه‌ی بهتر

1. اول اجرای آزمایشی با `--max-per-class 300` را امتحان کن تا مطمئن شوی مسیر دیتاست درست است.
2. برای آموزش اصلی از تمام داده‌ها استفاده کن و مدل‌ها را با **test accuracy و F1** مقایسه کن؛ فقط به دقت train تکیه نکن.
3. اگر train accuracy بالا رفت ولی validation accuracy ثابت ماند یا افت کرد، مدل احتمالاً بیش‌برازش کرده است؛ افزایش داده، توقف زودهنگام و وزن‌های ازپیش‌آموزش‌دیده کمک می‌کنند.
4. احتمال خروجی مدل لزوماً به معنی احتمال واقعی نیست و عکس‌های نامشخص یا چندحیوانی می‌توانند اشتباه طبقه‌بندی شوند.
