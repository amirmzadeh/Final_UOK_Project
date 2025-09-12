from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
from PIL import Image, ImageTk
import os

class YOLO_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Object Detection")
        self.root.geometry("800x600")
        
        # مدل پیش فرض
        self.model = None
        self.selected_model_path = ""
        
        # متغیرهای پردازش ویدیو
        self.cap = None
        self.is_webcam_active = False
        self.is_video_active = False
        
        self.create_widgets()
        
    def create_widgets(self):
        # فریم برای انتخاب مدل
        model_frame = ttk.LabelFrame(self.root, text="Model Selection", padding=10)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        # لیست مدل‌های موجود
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, sticky="w", pady=5)
        
        self.model_var = tk.StringVar()
        model_options = [
            "PPE Detection (Hardhat, Mask, Vest)",
            "Alpaca Detection",
            "Bee and Butterfly Detection (60 epochs)",
            "Ant and Insect Detection (5 epochs)",
            "Ant and Insect Detection (45 epochs)"
        ]
        
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=model_options, state="readonly")
        self.model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_select)
        
        # دکمه بارگذاری مدل
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=0, column=2, padx=5, pady=5)
        
        # فریم برای انتخاب منبع
        source_frame = ttk.LabelFrame(self.root, text="Source Selection", padding=10)
        source_frame.pack(fill="x", padx=10, pady=5)
        
        # دکمه‌های انتخاب منبع
        ttk.Button(source_frame, text="Select Image", command=self.select_image).pack(side="left", padx=5, pady=5)
        ttk.Button(source_frame, text="Select Video", command=self.select_video).pack(side="left", padx=5, pady=5)
        ttk.Button(source_frame, text="Start Webcam", command=self.start_webcam).pack(side="left", padx=5, pady=5)
        ttk.Button(source_frame, text="Stop", command=self.stop_processing).pack(side="left", padx=5, pady=5)
        
        # فریم برای نمایش نتایج
        result_frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # برچسب برای نمایش تصویر
        self.image_label = ttk.Label(result_frame)
        self.image_label.pack(fill="both", expand=True)
        
        # نوار وضعیت
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
    
    def on_model_select(self, event):
        model_mapping = {
            "PPE Detection (Hardhat, Mask, Vest)": "C:/Users/My Dell/YOLOv8-custom-object-detection/YOLOv8-cutom-object-detection/PPE-custom-object-detection-with-YOLOv8/ppe.pt",
            "Alpaca Detection": "C:/Users/My Dell/YOLOv8-custom-object-detection/YOLOv8-custom-object-detection/Custom-object-detection-with-YOLOv8/alpaca training results/weights/best.pt",
            "Bee and Butterfly Detection (60 epochs)": "C:/Users/My Dell/YOLOv8-custom-object-detection/Custom-object-detection-with-YOLOv8/Bee and Butterfly 60 epochs/weights/best.pt",
            "Ant and Insect Detection (5 epochs)": "C:/Users/My Dell/YOLOv8-custom-object-detection/YOLOv8-custom-object-detection/Custom-object-detection-with-YOLOv8/Ant and insect training results  5 epochs/weights/best.pt",
            "Ant and Insect Detection (45 epochs)": "C:/Users/My Dell/YOLOv8-custom-object-detection/YOLOv8-custom-object-detection/Custom-object-detection-with-YOLOv8/Ant and insect training results  45 epochs/weights/best.pt"
        }
        
        selected = self.model_var.get()
        self.selected_model_path = model_mapping.get(selected, "")
        self.status_var.set(f"Selected model: {selected}")
    
    def load_model(self):
        if not self.selected_model_path:
            messagebox.showerror("Error", "Please select a model first")
            return
        
        try:
            self.status_var.set("Loading model...")
            self.model = YOLO(self.selected_model_path)
            self.status_var.set(f"Model loaded successfully: {self.model_var.get()}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model")
    
    def select_image(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.stop_processing()
            self.process_image(file_path)
    
    def process_image(self, image_path):
        try:
            self.status_var.set("Processing image...")
            results = self.model.predict(source=image_path, save=True)
            
            # نمایش نتایج
            for r in results:
                detected_objects = [self.model.names[int(cls)] for cls in r.boxes.cls]
                confidences = [f"{conf:.2f}" for conf in r.boxes.conf]
                result_text = f"Detected: {list(zip(detected_objects, confidences))}"
                self.status_var.set(result_text)
            
            # نمایش تصویر پردازش شده
            result_image_path = os.path.join("runs", "detect", "predict", os.path.basename(image_path))
            if os.path.exists(result_image_path):
                self.display_image(result_image_path)
            
            self.status_var.set("Image processing completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.status_var.set("Error processing image")
    
    def select_video(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            self.stop_processing()
            self.is_video_active = True
            threading.Thread(target=self.process_video, args=(file_path,), daemon=True).start()
    
    def process_video(self, video_path):
        try:
            self.status_var.set("Processing video...")
            self.cap = cv2.VideoCapture(video_path)
            
            while self.is_video_active and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # پردازش فریم
                results = self.model(frame)
                annotated_frame = results[0].plot()
                
                # نمایش فریم
                self.display_frame(annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            self.cap.release()
            self.is_video_active = False
            self.status_var.set("Video processing completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process video: {str(e)}")
            self.status_var.set("Error processing video")
    
    def start_webcam(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        self.stop_processing()
        self.is_webcam_active = True
        threading.Thread(target=self.process_webcam, daemon=True).start()
    
    def process_webcam(self):
        try:
            self.status_var.set("Starting webcam...")
            self.cap = cv2.VideoCapture(0)
            
            while self.is_webcam_active and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # پردازش فریم
                results = self.model(frame)
                annotated_frame = results[0].plot()
                
                # نمایش فریم
                self.display_frame(annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            self.cap.release()
            self.is_webcam_active = False
            self.status_var.set("Webcam stopped")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {str(e)}")
            self.status_var.set("Error starting webcam")
    
    def display_image(self, image_path):
        try:
            image = Image.open(image_path)
            # تغییر اندازه تصویر برای نمایش
            image.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def display_frame(self, frame):
        try:
            # تبدیل فریم از BGR به RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            # تغییر اندازه تصویر برای نمایش
            image.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def stop_processing(self):
        self.is_webcam_active = False
        self.is_video_active = False
        if self.cap is not None:
            self.cap.release()
        self.status_var.set("Processing stopped")

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLO_GUI(root)
    root.mainloop()
