from ultralytics import YOLO

"""
مدل‌های موجود در این ریپوزیتوری:

1. مدل تشخیص تجهیزات محافظت شخصی (PPE):
   - مسیر: YOLOv8-custom-object-detection/PPE-cutom-object-detection-with-YOLOv8/ppe.pt
   - کلاس‌ها: Person, Hardhat, Mask, Safety Vest, NO-Hardhat, NO-Mask, NO-Safety Vest
   - کاربرد: تشخیص افراد و تجهیزات ایمنی آنها

2. مدل تشخیص آلپاکا (تک کلاسه):
   - مسیر: YOLOv8-custom-object-detection/Custom-object-detection-with-YOLOv8/alpaca training results/weights/best.pt
   - کلاس‌ها: Alpaca
   - کاربرد: تشخیص آلپاکا در تصاویر و ویدیوها

3. مدل تشخیص زنبور و پروانه (دو کلاسه):
   - مسیر: YOLOv8-custom-object-detection/Custom-object-detection-with-YOLOv8/Bee and Butterfly 60 epochs/weights/best.pt
   - کلاس‌ها: Bee, Butterfly
   - کاربرد: تشخیص زنبور و پروانه
   - تعداد اپوک‌های آموزش: 60

4. مدل تشخیص مورچه و حشرات (دو کلاسه) - نسخه 5 اپوک:
   - مسیر: YOLOv8-custom-object-detection/Custom-object-detection-with-YOLOv8/Ant and insect training results  5 epochs/weights/best.pt
   - کلاس‌ها: Ant, Insect
   - کاربرد: تشخیص مورچه و حشرات
   - تعداد اپوک‌های آموزش: 5

5. مدل تشخیص مورچه و حشرات (دو کلاسه) - نسخه 45 اپوک:
   - مسیر: YOLOv8-custom-object-detection/Custom-object-detection-with-YOLOv8/Ant and insect training results  45 epochs/weights/best.pt
   - کلاس‌ها: Ant, Insect
   - کاربرد: تشخیص مورچه و حشرات
   - تعداد اپوک‌های آموزش: 45
   - نکته: این مدل به دلیل تعداد اپوک بیشتر، احتمالاً عملکرد بهتری نسبت به نسخه 5 اپوک دارد

نکات مهم:
- برای هر مدل دو فایل وجود دارد: best.pt (بهترین عملکرد) و last.pt (آخرین ذخیره)
- توصیه می‌شود از فایل best.pt استفاده شود
- مسیر کامل مدل‌ها باید با '/home/abtin/project/yolo-tmp/' شروع شود
"""

# انتخاب مدل مورد نظر با برداشتن کامنت
# مدل تشخیص تجهیزات محافظت شخصی (PPE) - تشخیص کلاه ایمنی، ماسک و جلیقه ایمنی
# model = YOLO('/home/abtin/project/yolo-tmp/YOLOv8-custom-object-detection/PPE-cutom-object-detection-with-YOLOv8/ppe.pt')

# مدل تشخیص آلپاکا
# model = YOLO('/home/abtin/project/yolo-tmp/YOLOv8-custom-object-detection/Custom-object-detection-with-YOLOv8/alpaca training results/weights/best.pt')

# مدل تشخیص زنبور و پروانه (60 اپوک)
model = YOLO('C:/Users/My Dell/YOLOv8-custom-object-detection/Custom-object-detection-with-YOLOv8/Bee and Butterfly 60 epochs/weights/best.pt')

# مدل تشخیص مورچه و حشرات (5 اپوک)
# model = YOLO('/home/abtin/project/yolo-tmp/YOLOv8-custom-object-detection/Custom-object-detection-with-YOLOv8/Ant and insect training results  5 epochs/weights/best.pt')

# مدل تشخیص مورچه و حشرات (45 اپوک - دقت بیشتر)
# model = YOLO('/home/abtin/project/yolo-tmp/YOLOv8-custom-object-detection/Custom-object-detection-with-YOLOv8/Ant and insect training results  45 epochs/weights/best.pt')

# برای پیش‌بینی روی یک تصویر
results = model.predict(source='C:/Users/My Dell/YOLOv8-custom-object-detection/Zanbor.jpg', save=True)

# برای پیش‌بینی روی یک ویدیو
# results = model.predict(source='path/to/your/video.mp4')

# برای استفاده از وبکم
#results = model.predict(source=0, show=True)  # 0 برای وبکم پیش‌فرض

# برای نمایش نتایج
for r in results:
    print("Detected objects:", r.boxes.cls)  # کلاس‌های شناسایی شده
    print("Confidence scores:", r.boxes.conf)  # امتیاز اطمینان
