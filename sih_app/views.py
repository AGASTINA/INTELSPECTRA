from django.shortcuts import render, redirect, get_object_or_404
from sih_app.models import Event, RtspTable, FaceRecognitionModel, FaceDatabase, DetectionLog, AIModelEnabled
from django.http import JsonResponse, StreamingHttpResponse
import time
import onnxruntime as ort
import cv2
import threading
from datetime import datetime
from django.core.files.base import ContentFile
import io
import numpy as np
import json
import face_recognition
import os
from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.middleware.csrf import get_token
import onnxruntime as ort


#HOME
def home(request):
    if request.method == 'POST':
        try:
            title = request.POST.get('title')
            description = request.POST.get('description')
            location = request.POST.get('location')
            event_type = request.POST.get('event_type')
            
            print("Received form data:", title, description, location, event_type) 
            
            event = Event.objects.create(
                event_name=title,
                event_description=description,
                event_location=location,
                event_type=event_type
            )
            
            print("Event created:", event.id)  
            return redirect('home')
            
        except Exception as e:
            print("Error:", str(e))  
            return render(request, 'home.html', {
                'events': Event.objects.all().order_by('-event_created_date'),
                'error': str(e)
            })
    
    events = Event.objects.all().order_by('-event_created_date')
    return render(request, 'home.html', {'events': events})

def edit_event(request, event_id):
    if request.method == 'POST':
        try:
            event = get_object_or_404(Event, id=event_id)
            event.event_name = request.POST.get('title')
            event.event_description = request.POST.get('description')
            event.event_location = request.POST.get('location')
            event.event_type = request.POST.get('event_type')
            event.save()
            print("Event updated:", event.id)
            return redirect('home')
        except Exception as e:
            print("Error updating:", str(e))
            return redirect('home')
    
    event = get_object_or_404(Event, id=event_id)
    return render(request, 'home.html', {
        'events': Event.objects.all().order_by('-event_created_date'),
        'edit_event': event
    })


def login_view(request):
    """Simple login endpoint. GET serves the login page and ensures the CSRF cookie is set.
    POST accepts JSON or form-encoded credentials and returns JSON with success/failure.
    """
    # Ensure CSRF cookie is present for client-side JS to read
    if request.method == 'GET':
        try:
            get_token(request)
        except Exception:
            pass
        return render(request, 'login.html')

    # POST - accept form-encoded or JSON
    if request.method == 'POST':
        try:
            # prefer POST form data
            username = request.POST.get('username')
            password = request.POST.get('password')

            # fallback to JSON body
            if not username:
                try:
                    body = json.loads(request.body.decode('utf-8') or '{}')
                    username = body.get('username')
                    password = body.get('password')
                except Exception:
                    username = None

            if not username or not password:
                return JsonResponse({'status': 'error', 'message': 'Missing credentials'}, status=400)

            user = authenticate(request, username=username, password=password)
            if user is not None:
                auth_login(request, user)
                return JsonResponse({'status': 'success', 'redirect': '/'})
            else:
                return JsonResponse({'status': 'error', 'message': 'Invalid credentials'}, status=401)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def logout_view(request):
    """Log the user out and redirect to login page or return JSON if AJAX."""
    if request.method == 'POST' or request.method == 'GET':
        try:
            auth_logout(request)
            # If AJAX, return JSON, otherwise redirect
            if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.is_ajax():
                return JsonResponse({'status': 'success', 'redirect': '/login'})
            return redirect('/login')
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def delete_event(request, event_id):
    if request.method == 'POST':
        try:
            event = get_object_or_404(Event, id=event_id)
            event.delete()
            print("Event deleted:", event_id)
        except Exception as e:
            print("Error deleting:", str(e))
    return redirect('home')

# EVENT PAGE
def event(request, event_id):
    event = get_object_or_404(Event, id=event_id)
    cameras = RtspTable.objects.filter(event_name=event)
    pinned_cameras = cameras.filter(is_pinned=True)[:2]
    normal_cameras = cameras.filter(is_pinned=False)  
    
    return render(request, 'event.html', {
        'event': event,
        'pinned_cameras': pinned_cameras,
        'normal_cameras': normal_cameras,
        'total_cameras': cameras.count()
    })


def face_recognition_page(request, event_id):
    """Face Recognition management page"""
    event = get_object_or_404(Event, id=event_id)
    return render(request, 'face_recognition.html', {'event': event})


#RTSP CONNECTION MANAGEMENT

camera_streams = {}

# Cache face encodings per event_id to avoid reloading from disk every frame
face_encodings_cache = {}

# Fire detection ONNX model initialization
fire_model_path = os.path.join(settings.BASE_DIR, 'static', 'ai_models', 'fire_v1.onnx')
try:
    fire_onnx_session = ort.InferenceSession(fire_model_path, providers=['CPUExecutionProvider'])
    fire_model_input_name = fire_onnx_session.get_inputs()[0].name
    fire_model_output_name = fire_onnx_session.get_outputs()[0].name
    print(f"Fire detection model loaded successfully from {fire_model_path}")
except Exception as e:
    fire_onnx_session = None
    print(f"Warning: Could not load fire detection model: {e}")

class FireDetectionManager:
    """Manages fire detection using ONNX model with performance optimizations"""
    def __init__(self):
        self.session = fire_onnx_session
        self.input_name = fire_model_input_name if fire_onnx_session else None
        self.output_name = fire_model_output_name if fire_onnx_session else None
        self.input_size = (416, 416)  # Reduced for consistency and performance
        self.conf_threshold = 0.6  # Higher threshold for better accuracy
        self.iou_threshold = 0.45
        self.max_detections = 5  # Limit fire detections processed
    
    def preprocess(self, frame):
        """Preprocess frame for ONNX model inference"""
        # Resize to model input size
        img = cv2.resize(frame, self.input_size)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1] and change to CHW format
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    
    def postprocess(self, outputs, orig_shape):
        """Process model outputs to get bounding boxes"""
        if outputs is None or len(outputs) == 0:
            return []
        
        predictions = outputs[0]  # Get first output
        
        # Handle different output formats
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        
        detections = []
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        
        for pred in predictions:
            if len(pred) < 6:
                continue
            
            # Extract detection info: [x1, y1, x2, y2, confidence, class_id]
            x1, y1, x2, y2, conf, cls = pred[:6]
            
            if conf < self.conf_threshold:
                continue
            
            # Scale coordinates back to original image size
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'class_id': int(cls),
                'label': 'Fire'
            })
        
        return detections
    
    def detect_fire(self, frame):
        """Detect fire in frame"""
        if self.session is None:
            return []
        
        try:
            # Preprocess frame
            input_tensor = self.preprocess(frame)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            
            # Postprocess outputs
            detections = self.postprocess(outputs, frame.shape)
            
            return detections
        except Exception as e:
            print(f"Fire detection error: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['label']
            
            # Draw bounding box in red for fire
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Prepare label text
            text = f"{label}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw red background for text
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                        (x1 + text_width + 6, y1), (0, 0, 255), -1)
            
            # Draw text in white
            cv2.putText(frame, text, (x1 + 3, y1 - 5), 
                      font, font_scale, (255, 255, 255), thickness)
        
        return frame

class WeaponDetectionManager:
    """Manages weapon detection using ONNX model with performance optimizations"""
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_size = (416, 416)  # Reduced from 640x640 for better performance
        self.conf_threshold = 0.65  # Slightly higher threshold to reduce false positives
        self.iou_threshold = 0.45
        self.load_model()
        
        # Performance optimization
        self.skip_frames = 0
        self.max_detections = 10  # Limit number of detections processed
    
    def load_model(self):
        """Load weapon detection model"""
        try:
            model_path = os.path.join(settings.BASE_DIR, 'static', 'ai_models', 'weapon_v1.onnx')
            if os.path.exists(model_path):
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                print(f"Weapon detection model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weapon detection model: {e}")
    
    def preprocess(self, frame):
        """Preprocess frame for weapon detection"""
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def detect_weapons(self, frame):
        """Detect weapons in frame"""
        if self.session is None:
            return []
        
        try:
            input_tensor = self.preprocess(frame)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            detections = self.postprocess(outputs, frame.shape)
            return detections
        except Exception as e:
            print(f"Weapon detection error: {e}")
            return []
    
    def postprocess(self, outputs, orig_shape):
        """Process weapon detection outputs"""
        if outputs is None or len(outputs) == 0:
            return []
        
        predictions = outputs[0]
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        
        detections = []
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        
        for pred in predictions:
            if len(pred) < 6:
                continue
            
            x1, y1, x2, y2, conf, cls = pred[:6]
            
            if conf < self.conf_threshold:
                continue
            
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'class_id': int(cls),
                'label': 'Weapon'
            })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw weapon detection bounding boxes"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['label']
            
            # Draw bounding box in red for weapons
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            text = f"{label}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                        (x1 + text_width + 6, y1), (0, 0, 255), -1)
            
            cv2.putText(frame, text, (x1 + 3, y1 - 5), 
                      font, font_scale, (255, 255, 255), thickness)
        
        return frame

class VehicleDetectionManager:
    """Manages vehicle detection using ONNX model with performance optimizations"""
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_size = (416, 416)  # Reduced for better performance
        self.conf_threshold = 0.6  # Higher threshold for fewer false positives
        self.iou_threshold = 0.45
        self.load_model()
        
        # Performance optimization
        self.max_detections = 15  # Limit detections processed
    
    def load_model(self):
        """Load vehicle detection model"""
        try:
            model_path = os.path.join(settings.BASE_DIR, 'static', 'ai_models', 'vechicle_v1.onnx')
            if os.path.exists(model_path):
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                print(f"Vehicle detection model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load vehicle detection model: {e}")
    
    def preprocess(self, frame):
        """Preprocess frame for vehicle detection"""
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def detect_vehicles(self, frame):
        """Detect vehicles in frame"""
        if self.session is None:
            return []
        
        try:
            input_tensor = self.preprocess(frame)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            detections = self.postprocess(outputs, frame.shape)
            return detections
        except Exception as e:
            print(f"Vehicle detection error: {e}")
            return []
    
    def postprocess(self, outputs, orig_shape):
        """Process vehicle detection outputs"""
        if outputs is None or len(outputs) == 0:
            return []
        
        predictions = outputs[0]
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        
        detections = []
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        
        for pred in predictions:
            if len(pred) < 6:
                continue
            
            x1, y1, x2, y2, conf, cls = pred[:6]
            
            if conf < self.conf_threshold:
                continue
            
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            # Map class IDs to vehicle types
            vehicle_types = {0: 'Car', 1: 'Truck', 2: 'Bus', 3: 'Motorcycle', 4: 'Bicycle'}
            label = vehicle_types.get(int(cls), 'Vehicle')
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'class_id': int(cls),
                'label': label
            })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw vehicle detection bounding boxes"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['label']
            
            # Draw bounding box in blue for vehicles
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            text = f"{label}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                        (x1 + text_width + 6, y1), (255, 0, 0), -1)
            
            cv2.putText(frame, text, (x1 + 3, y1 - 5), 
                      font, font_scale, (255, 255, 255), thickness)
        
        return frame

class SuspiciousActivityManager:
    """Manages suspicious activity detection using NSG model with performance optimizations"""
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_size = (416, 416)  # Reduced for better performance
        self.conf_threshold = 0.6  # Higher threshold for better accuracy
        self.iou_threshold = 0.45
        self.load_model()
        
        # Performance optimization
        self.max_detections = 8  # Limit suspicious activity detections
    
    def load_model(self):
        """Load suspicious activity detection model"""
        try:
            model_path = os.path.join(settings.BASE_DIR, 'static', 'ai_models', 'nsg_v1.onnx')
            if os.path.exists(model_path):
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                print(f"Suspicious activity detection model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load suspicious activity detection model: {e}")
    
    def preprocess(self, frame):
        """Preprocess frame for suspicious activity detection"""
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def detect_suspicious_activity(self, frame):
        """Detect suspicious activities in frame"""
        if self.session is None:
            return []
        
        try:
            input_tensor = self.preprocess(frame)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            detections = self.postprocess(outputs, frame.shape)
            return detections
        except Exception as e:
            print(f"Suspicious activity detection error: {e}")
            return []
    
    def postprocess(self, outputs, orig_shape):
        """Process suspicious activity detection outputs"""
        if outputs is None or len(outputs) == 0:
            return []
        
        predictions = outputs[0]
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        
        detections = []
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        
        for pred in predictions:
            if len(pred) < 6:
                continue
            
            x1, y1, x2, y2, conf, cls = pred[:6]
            
            if conf < self.conf_threshold:
                continue
            
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            # Map class IDs to suspicious activities
            activity_types = {
                0: 'Loitering', 1: 'Trespassing', 2: 'Suspicious Movement', 
                3: 'Abandoned Object', 4: 'Crowd Formation', 5: 'Fighting'
            }
            label = activity_types.get(int(cls), 'Suspicious Activity')
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'class_id': int(cls),
                'label': label
            })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw suspicious activity detection bounding boxes"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['label']
            
            # Draw bounding box in orange for suspicious activities
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            
            text = f"{label}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                        (x1 + text_width + 6, y1), (0, 165, 255), -1)
            
            cv2.putText(frame, text, (x1 + 3, y1 - 5), 
                      font, font_scale, (255, 255, 255), thickness)
        
        return frame

def get_face_encodings_for_event(event_id):
    """Load face encodings from FaceDatabase for the event and cache them.

    Returns list of tuples: (person_name, encoding)
    """
    try:
        if event_id in face_encodings_cache:
            return face_encodings_cache[event_id]

        enc_list = []
        faces = FaceDatabase.objects.filter(event_name_id=event_id)
        for face in faces:
            try:
                # Build absolute path to the stored image
                img_path = os.path.join(settings.MEDIA_ROOT, face.face_image.name)
                if not os.path.exists(img_path):
                    continue
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if len(encs) > 0:
                    enc_list.append((face.person_name, encs[0]))
            except Exception:
                continue

        face_encodings_cache[event_id] = enc_list
        return enc_list
    except Exception:
        return []

def add_rtsp_camera(request, event_id):
    """Add a new camera/stream to an event - supports multiple protocols"""
    if request.method == 'POST':
        try:
            event = get_object_or_404(Event, id=event_id)
            camera_name = request.POST.get('camera_name')
            rtsp_url = request.POST.get('rtsp_url')
            stream_type = request.POST.get('stream_type', 'rtsp')
            camera_location = request.POST.get('camera_location', '')
            
            # Validate and process URL based on stream type
            if stream_type == 'usb':
                # For USB cameras, expect device index (0, 1, 2, etc.)
                try:
                    device_index = int(rtsp_url)
                    rtsp_url = str(device_index)
                except ValueError:
                    rtsp_url = '0'  # Default to first USB camera
            
            # Create camera entry
            rtsp_camera = RtspTable.objects.create(
                event_name=event,
                camera_name=camera_name,
                rtsp_url=rtsp_url,
                stream_type=stream_type
            )
            
            print(f"RTSP Camera added: {camera_name} - {rtsp_url}")
            
            return JsonResponse({
                'status': 'success',
                'camera_id': rtsp_camera.id,
                'message': 'Camera added successfully'
            })
            
        except Exception as e:
            print(f"Error adding camera: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def delete_rtsp_camera(request, camera_id):
    """Delete an RTSP camera"""
    if request.method == 'POST':
        try:
            camera = get_object_or_404(RtspTable, id=camera_id)
            camera.delete()
            
            # Stop stream if active and cleanup properly
            if camera_id in camera_streams:
                try:
                    camera_streams[camera_id].stop()
                except Exception:
                    pass
                del camera_streams[camera_id]
            
            return JsonResponse({
                'status': 'success',
                'message': 'Camera deleted successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def get_event_cameras(request, event_id):
    """Get all cameras for an event"""
    try:
        event = get_object_or_404(Event, id=event_id)
        cameras = RtspTable.objects.filter(event_name=event).values(
            'id', 'camera_name', 'rtsp_url', 'stream_type', 'added_at'
        )
        
        return JsonResponse({
            'status': 'success',
            'cameras': list(cameras)
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)

def test_rtsp_connection(request):
    """Test stream connection before adding - supports all protocols"""
    if request.method == 'POST':
        try:
            rtsp_url = request.POST.get('rtsp_url')
            stream_type = request.POST.get('stream_type', 'rtsp')
            
            # Handle USB camera
            if stream_type == 'usb':
                try:
                    device_index = int(rtsp_url)
                except ValueError:
                    device_index = 0
                cap = cv2.VideoCapture(device_index)
            else:
                # Try to connect to network stream or file
                cap = cv2.VideoCapture(rtsp_url)
            
            success = cap.isOpened()
            
            if success:
                # Read one frame to verify
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    return JsonResponse({
                        'status': 'success',
                        'message': f'{stream_type.upper()} connection successful'
                    })
            
            cap.release()
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to connect to {stream_type.upper()} stream'
            }, status=400)
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def pin_camera(request, camera_id):
    """Pin a camera to the pinned screens"""
    if request.method == 'POST':
        try:
            camera = get_object_or_404(RtspTable, id=camera_id)
            
            # Check if already 2 pinned cameras for this event
            pinned_count = RtspTable.objects.filter(
                event_name=camera.event_name, 
                is_pinned=True
            ).count()
            
            if pinned_count >= 2:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Maximum 2 cameras can be pinned'
                }, status=400)
            
            camera.is_pinned = True
            camera.save()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Camera pinned successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def unpin_camera(request, camera_id):
    """Unpin a camera from pinned screens"""
    if request.method == 'POST':
        try:
            camera = get_object_or_404(RtspTable, id=camera_id)
            camera.is_pinned = False
            camera.save()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Camera unpinned successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

class VideoCamera:
    """Class to handle video streaming from multiple sources with AI detection"""
    def __init__(self, rtsp_url, camera_id, event_id, enable_detection=True, stream_type='rtsp'):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.event_id = event_id
        self.stream_type = stream_type
        self.enable_detection = enable_detection
        self.video = None
        self.lock = threading.Lock()
        self.stopped = False
        self.frame_count = 0
        self.detection_manager = None
        self.last_detections = {}  # Store last detections to draw on all frames
        self.last_face_annotations = []  # Store last face annotations (list of ((top,right,bottom,left), name))
        self.face_recognition_mode = False  # when True, only show face bounding boxes and skip other detection overlays
        self.fire_detection_mode = False  # when True, enable fire detection
        self.weapon_detection_mode = False  # when True, enable weapon detection
        self.vehicle_detection_mode = False  # when True, enable vehicle detection
        self.suspicious_detection_mode = False  # when True, enable suspicious activity detection
        self._ai_flags_last_checked = 0
        self.fire_detector = FireDetectionManager()
        self.weapon_detector = WeaponDetectionManager()
        self.vehicle_detector = VehicleDetectionManager()
        self.suspicious_detector = SuspiciousActivityManager()
        
        self.last_fire_detections = []  # Store last fire detections
        self.last_weapon_detections = []  # Store last weapon detections
        self.last_vehicle_detections = []  # Store last vehicle detections
        self.last_suspicious_detections = []  # Store last suspicious activity detections
        
        self.last_fire_alert_time = 0  # Track when last fire alert was sent
        self.last_weapon_alert_time = 0  # Track when last weapon alert was sent
        self.latest_frame = None
        self.stopped = False
        
        # Performance optimization for multiple models
        self.max_concurrent_detections = 2  # Limit concurrent AI processing
        self.detection_queue = []
        self.current_detections_running = 0
        self.frame_skip_factor = 1  # Dynamic frame skipping based on load
        self.performance_mode = 'balanced'  # 'fast', 'balanced', 'accurate'
        
        # Face tracking variables for smooth stable recognition
        self.tracked_faces = {}  # {track_id: {'bbox': (t,r,b,l), 'name': str, 'confidence': float, 'frames_tracked': int}}
        self.next_track_id = 0
        self.max_tracking_distance = 80  # Maximum pixel distance to consider same face
        self.min_confidence_frames = 3  # Require 3 consistent frames before showing name
        self.max_lost_frames = 10  # Remove track after this many frames without detection
        
        self._recognition_thread = threading.Thread(target=self._recognition_worker, daemon=True)
        self.connect()
        # Start recognition worker
        try:
            self._recognition_thread.start()
        except Exception:
            pass

    def refresh_ai_flags(self):
        """Refresh AI flags from DB for this event. Cheap check cached per seconds/frames."""
        try:
            cfg = AIModelEnabled.objects.filter(target_event_id=self.event_id).first()
            if cfg:
                self.face_recognition_mode = bool(cfg.is_face_recognition_enabled)
                self.fire_detection_mode = bool(cfg.is_fire_detection_enabled)
                self.weapon_detection_mode = bool(cfg.is_weapon_detection_enabled)
                self.vehicle_detection_mode = bool(cfg.is_vechicle_detection_enabled)  # Note: field has typo in model
                self.suspicious_detection_mode = bool(cfg.is_suspicious_detection_enabled)
            else:
                self.face_recognition_mode = False
                self.fire_detection_mode = False
                self.weapon_detection_mode = False
                self.vehicle_detection_mode = False
                self.suspicious_detection_mode = False
        except Exception:
            self.face_recognition_mode = False
            self.fire_detection_mode = False
            self.weapon_detection_mode = False
            self.vehicle_detection_mode = False
            self.suspicious_detection_mode = False
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        # box format: (top, right, bottom, left)
        t1, r1, b1, l1 = box1
        t2, r2, b2, l2 = box2
        
        # Calculate intersection
        x_left = max(l1, l2)
        y_top = max(t1, t2)
        x_right = min(r1, r2)
        y_bottom = min(b1, b2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (r1 - l1) * (b1 - t1)
        box2_area = (r2 - l2) * (b2 - t2)
        
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def calculate_distance(self, box1, box2):
        """Calculate center distance between two bounding boxes"""
        # box format: (top, right, bottom, left)
        t1, r1, b1, l1 = box1
        t2, r2, b2, l2 = box2
        
        # Calculate centers
        c1_x = (l1 + r1) / 2
        c1_y = (t1 + b1) / 2
        c2_x = (l2 + r2) / 2
        c2_y = (t2 + b2) / 2
        
        # Euclidean distance
        distance = ((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2) ** 0.5
        return distance
    
    def update_tracked_faces(self, new_detections):
        """Update face tracking with new detections
        
        new_detections: list of ((top, right, bottom, left), name) tuples
        Returns: list of ((top, right, bottom, left), name, track_id) with stable tracking
        """
        current_time = self.frame_count
        
        # Mark all tracks as not updated
        for track_id in self.tracked_faces:
            self.tracked_faces[track_id]['updated'] = False
        
        matched_detections = []
        
        # Match new detections to existing tracks
        for bbox, name in new_detections:
            best_match_id = None
            best_match_score = 0
            
            # Find best matching track based on IOU and distance
            for track_id, track_info in self.tracked_faces.items():
                iou = self.calculate_iou(bbox, track_info['bbox'])
                distance = self.calculate_distance(bbox, track_info['bbox'])
                
                # Use IOU for matching (more robust than distance)
                if iou > 0.3 and distance < self.max_tracking_distance:
                    score = iou
                    if score > best_match_score:
                        best_match_score = score
                        best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                track = self.tracked_faces[best_match_id]
                track['bbox'] = bbox
                track['last_seen'] = current_time
                track['frames_tracked'] += 1
                track['updated'] = True
                
                # Update name with voting mechanism for stability
                if name == track['name']:
                    track['name_confidence'] = min(track['name_confidence'] + 1, 10)
                else:
                    track['name_confidence'] -= 1
                    if track['name_confidence'] <= 0:
                        track['name'] = name
                        track['name_confidence'] = 1
                
                # Only show name if tracked for minimum frames
                display_name = track['name'] if track['frames_tracked'] >= self.min_confidence_frames else "Detecting..."
                matched_detections.append((bbox, display_name, best_match_id))
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracked_faces[track_id] = {
                    'bbox': bbox,
                    'name': name,
                    'name_confidence': 1,
                    'frames_tracked': 1,
                    'last_seen': current_time,
                    'updated': True
                }
                
                # Show "Detecting..." for new faces
                display_name = "Detecting..." if self.min_confidence_frames > 1 else name
                matched_detections.append((bbox, display_name, track_id))
        
        # Remove tracks that haven't been updated for too long
        tracks_to_remove = []
        for track_id, track_info in self.tracked_faces.items():
            if not track_info.get('updated', False):
                if current_time - track_info['last_seen'] > self.max_lost_frames:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_faces[track_id]
        
        return matched_detections
    
    def connect(self):
        """Connect to video stream - supports multiple protocols"""
        try:
            # Handle different stream types
            if self.stream_type == 'usb':
                # USB camera - convert to integer device index
                try:
                    device_index = int(self.rtsp_url)
                except ValueError:
                    device_index = 0
                    print(f"Invalid USB device index '{self.rtsp_url}', defaulting to 0")
                
                print(f"Attempting to connect to USB camera at index {device_index}...")
                self.video = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)  # Use DirectShow on Windows for better compatibility
                
                if self.video.isOpened():
                    print(f"Successfully connected to USB camera {device_index}")
                else:
                    self.video = None
                    print(f"FAILED: No USB camera found at index {device_index}. Please check:")
                    print(f"  1. Is a webcam physically connected?")
                    print(f"  2. Is it being used by another application?")
                    print(f"  3. Try different device indices (0, 1, 2...)")
                return  # Early return for USB to prevent fallback attempts
                
            elif self.stream_type == 'file':
                # Video file path
                print(f"Attempting to open video file: {self.rtsp_url}")
                self.video = cv2.VideoCapture(self.rtsp_url)
                
            elif self.stream_type in ['rtsp', 'rtmp', 'http', 'hls', 'ip', 'other']:
                # Network streams - OpenCV handles these automatically
                print(f"Attempting to connect to {self.stream_type.upper()} stream: {self.rtsp_url}")
                self.video = cv2.VideoCapture(self.rtsp_url)
                
                # Set buffer size for network streams to reduce latency
                if self.stream_type in ['rtsp', 'rtmp']:
                    self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
            else:
                # Default fallback
                print(f"Attempting to connect to stream: {self.rtsp_url}")
                self.video = cv2.VideoCapture(self.rtsp_url)
            
            # Verify connection (for non-USB streams)
            if not self.video.isOpened():
                self.video = None
                print(f"Failed to open {self.stream_type} stream: {self.rtsp_url}")
            else:
                print(f"Successfully connected to {self.stream_type} stream")
                
        except Exception as e:
            print(f"Error connecting to stream: {e}")
            self.video = None
    
    def get_frame(self):
        """Get a frame from the video stream with optional AI detection"""
        if self.video is None or not self.video.isOpened():
            # For USB cameras that failed initially, don't spam reconnection attempts
            if self.stream_type == 'usb' and self.video is None and self.frame_count > 0:
                return None  # Already tried and failed
            
            self.connect()
            if self.video is None:
                return None
        
        with self.lock:
            success, frame = self.video.read()
            if not success:
                self.connect()
                return None

            # store latest frame for background recognition worker (only if needed)
            try:
                if self.face_recognition_mode:
                    self.latest_frame = frame.copy()
                else:
                    self.latest_frame = None
            except Exception:
                self.latest_frame = None

            self.frame_count += 1
            
            # Adaptive frame skipping based on enabled detections
            active_detections = sum([
                self.fire_detection_mode,
                self.weapon_detection_mode, 
                self.vehicle_detection_mode,
                self.suspicious_detection_mode,
                self.face_recognition_mode
            ])
            
            # Increase frame skipping if multiple detections are active
            if active_detections > 2:
                self.frame_skip_factor = 2
            elif active_detections > 3:
                self.frame_skip_factor = 3
            else:
                self.frame_skip_factor = 1
            
            # Fire Detection: optimized intervals based on load
            try:
                # refresh AI flags less frequently when busy (every 120 frames)
                if self.frame_count % (120 * self.frame_skip_factor) == 0:
                    self.refresh_ai_flags()
                
                if self.fire_detection_mode and self.fire_detector and self.fire_detector.session:
                    if self.frame_count % (15 * self.frame_skip_factor) == 0:
                        try:
                            # Detect fire in frame
                            fire_detections = self.fire_detector.detect_fire(frame)
                            self.last_fire_detections = fire_detections
                            
                            # Log fire detection and create alert
                            if fire_detections and self.frame_count - self.last_fire_alert_time > 90:  # Alert every 3 seconds (90 frames)
                                self.last_fire_alert_time = self.frame_count
                                try:
                                    self._save_fire_detection_log(frame, fire_detections)
                                except Exception as e:
                                    print(f"Error saving fire detection log: {e}")
                        except Exception as e:
                            print(f"Fire detection error in get_frame: {e}")
                    
                    # Draw fire detections on frame
                    if self.last_fire_detections:
                        try:
                            frame = self.fire_detector.draw_detections(frame, self.last_fire_detections)
                        except Exception as e:
                            pass
                else:
                    # Clear fire detections if fire detection is disabled
                    self.last_fire_detections = []
            except Exception as e:
                print(f"Fire detection processing error: {e}")
            
            # Weapon Detection: optimized for performance
            try:
                if self.weapon_detection_mode and self.weapon_detector and self.weapon_detector.session:
                    # Reduce frequency when multiple detections are active
                    weapon_interval = 12 * self.frame_skip_factor
                    if self.frame_count % weapon_interval == 0:
                        try:
                            # Detect weapons in frame
                            weapon_detections = self.weapon_detector.detect_weapons(frame)
                            self.last_weapon_detections = weapon_detections
                            
                            # Log weapon detection and create alert
                            if weapon_detections and self.frame_count - self.last_weapon_alert_time > 60:  # Alert every 2 seconds (60 frames)
                                self.last_weapon_alert_time = self.frame_count
                                try:
                                    self._save_weapon_detection_log(frame, weapon_detections)
                                except Exception as e:
                                    print(f"Error saving weapon detection log: {e}")
                        except Exception as e:
                            print(f"Weapon detection error in get_frame: {e}")
                    
                    # Draw weapon detections on frame
                    if self.last_weapon_detections:
                        try:
                            frame = self.weapon_detector.draw_detections(frame, self.last_weapon_detections)
                        except Exception as e:
                            pass
                else:
                    # Clear weapon detections if weapon detection is disabled
                    self.last_weapon_detections = []
            except Exception as e:
                print(f"Weapon detection processing error: {e}")
            
            # Vehicle Detection: reduced frequency for performance
            try:
                if self.vehicle_detection_mode and self.vehicle_detector and self.vehicle_detector.session:
                    # Much less frequent processing for vehicles (non-critical)
                    vehicle_interval = 20 * self.frame_skip_factor
                    if self.frame_count % vehicle_interval == 0:
                        try:
                            # Detect vehicles in frame
                            vehicle_detections = self.vehicle_detector.detect_vehicles(frame)
                            self.last_vehicle_detections = vehicle_detections
                            
                            # Log vehicle detection (less frequent alerts - every 150 frames / 5 seconds)
                            if vehicle_detections and self.frame_count % 150 == 0:
                                try:
                                    self._save_vehicle_detection_log(frame, vehicle_detections)
                                except Exception as e:
                                    print(f"Error saving vehicle detection log: {e}")
                        except Exception as e:
                            print(f"Vehicle detection error in get_frame: {e}")
                    
                    # Draw vehicle detections on frame
                    if self.last_vehicle_detections:
                        try:
                            frame = self.vehicle_detector.draw_detections(frame, self.last_vehicle_detections)
                        except Exception as e:
                            pass
                else:
                    # Clear vehicle detections if vehicle detection is disabled
                    self.last_vehicle_detections = []
            except Exception as e:
                print(f"Vehicle detection processing error: {e}")
            
            # Suspicious Activity Detection: reduced frequency for performance
            try:
                if self.suspicious_detection_mode and self.suspicious_detector and self.suspicious_detector.session:
                    # Less frequent processing for suspicious activities
                    suspicious_interval = 25 * self.frame_skip_factor
                    if self.frame_count % suspicious_interval == 0:
                        try:
                            # Detect suspicious activities in frame
                            suspicious_detections = self.suspicious_detector.detect_suspicious_activity(frame)
                            self.last_suspicious_detections = suspicious_detections
                            
                            # Log suspicious activity detection (every 120 frames / 4 seconds)
                            if suspicious_detections and self.frame_count % 120 == 0:
                                try:
                                    self._save_suspicious_detection_log(frame, suspicious_detections)
                                except Exception as e:
                                    print(f"Error saving suspicious activity detection log: {e}")
                        except Exception as e:
                            print(f"Suspicious activity detection error in get_frame: {e}")
                    
                    # Draw suspicious activity detections on frame
                    if self.last_suspicious_detections:
                        try:
                            frame = self.suspicious_detector.draw_detections(frame, self.last_suspicious_detections)
                        except Exception as e:
                            pass
                else:
                    # Clear suspicious detections if suspicious detection is disabled
                    self.last_suspicious_detections = []
            except Exception as e:
                print(f"Suspicious activity detection processing error: {e}")
            
            # Run detection every 5 frames to save processing power
            if self.enable_detection and self.detection_manager:
                if self.frame_count % 5 == 0:
                    try:
                        # Run detection and update last detections
                        self.last_detections = self.detection_manager.detect_frame(frame)
                        
                        # Log detections every 30 frames (reduce database writes)
                        if self.last_detections and self.frame_count % 30 == 0:
                            self.log_detections(self.last_detections, frame)
                    except Exception as e:
                        pass
                
                # Draw the last detected bounding boxes on EVERY frame for smooth display
                if self.last_detections:
                    try:
                        frame = self.detection_manager.draw_detections(frame, self.last_detections)
                    except Exception as e:
                        pass

            # Face recognition: adaptive intervals based on system load
            try:
                face_interval = 8 * self.frame_skip_factor
                if self.frame_count % face_interval == 0:
                    try:
                        # If face recognition is not enabled for this event, skip processing
                        if not self.face_recognition_mode:
                            # ensure annotations cleared
                            self.last_face_annotations = []
                            self.tracked_faces.clear()  # Clear tracking when disabled
                            raise Exception("face recognition disabled")

                        # Convert to RGB for face_recognition library
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Detect face locations
                        locs = face_recognition.face_locations(rgb, model='hog')  # Use HOG for faster detection

                        raw_detections = []

                        if len(locs) > 0:
                            # Get face encodings and match with known faces in database
                            encs = face_recognition.face_encodings(rgb, locs)
                            known = get_face_encodings_for_event(self.event_id)
                            known_names = [k[0] for k in known]
                            known_vecs = [k[1] for k in known]

                            for (top, right, bottom, left), enc in zip(locs, encs):
                                name = "Unknown"  # Default label for unrecognized faces
                                if len(known_vecs) > 0:
                                    matches = face_recognition.compare_faces(known_vecs, enc, tolerance=0.25)
                                    if True in matches:
                                        idx = matches.index(True)
                                        name = known_names[idx]
                                raw_detections.append(((top, right, bottom, left), name))

                        # Apply tracking for smooth and stable recognition
                        self.last_face_annotations = self.update_tracked_faces(raw_detections)
                        
                        # Explicitly delete large objects to free memory
                        del rgb
                        if 'encs' in locals():
                            del encs
                    except Exception:
                        self.last_face_annotations = []                # Draw last face annotations on every frame, but only when face recognition is enabled
                # Only draw bounding boxes for RECOGNIZED faces (skip "Unknown" and "Detecting...")
                if self.face_recognition_mode and self.last_face_annotations:
                    try:
                        for item in self.last_face_annotations:
                            # Unpack based on tuple length (with or without track_id)
                            if len(item) == 3:
                                (top, right, bottom, left), name, track_id = item
                            else:
                                (top, right, bottom, left), name = item
                                track_id = None
                            
                            # Skip drawing if face is "Unknown" or "Detecting..."
                            if name == "Detecting..." or name == "Unknown" or not name:
                                continue
                            
                            # Log recognized face detection (every 30 frames to avoid spam)
                            if self.frame_count % 30 == 0:
                                try:
                                    self._save_detection_log(frame, name, (top, right, bottom, left))
                                except Exception:
                                    pass
                            
                            # Only draw for recognized faces (with actual names)
                            box_color = (0, 255, 0)  # Green for recognized
                            text_bg_color = (0, 255, 0)
                            
                            # Draw bounding box with thicker line for better visibility
                            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                            
                            # Display the person's name label
                            label = name
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.6
                            thickness = 2
                            
                            # Get text size for background rectangle
                            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                            
                            # Draw filled rectangle as background for text
                            cv2.rectangle(frame, (left, top - text_height - 10), 
                                        (left + text_width + 6, top), text_bg_color, -1)
                            
                            # Draw text label in black color on colored background
                            cv2.putText(frame, label, (left + 3, top - 5), 
                                      font, font_scale, (0, 0, 0), thickness)
                            
                            # Optional: Draw track ID for debugging (small corner indicator)
                            if track_id is not None and False:  # Set to True to enable
                                cv2.putText(frame, f"#{track_id}", (right - 30, top + 15),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
                    except Exception as e:
                        pass
            except Exception:
                pass
            
            # Dynamic JPEG quality based on system load
            jpeg_quality = 85
            if active_detections > 2:
                jpeg_quality = 70  # Reduce quality when busy
            elif active_detections > 3:
                jpeg_quality = 60  # Further reduce for heavy load
            
            # Encode frame as JPEG with adaptive quality
            ret, jpeg = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            if ret:
                return jpeg.tobytes()
            return None
    
    def stop(self):
        """Stop the video camera and cleanup resources"""
        self.stopped = True
        try:
            # Wait for recognition thread to finish
            if hasattr(self, '_recognition_thread') and self._recognition_thread.is_alive():
                self._recognition_thread.join(timeout=1.0)
        except Exception:
            pass
        
        try:
            if self.video is not None:
                self.video.release()
                self.video = None
        except Exception:
            pass
        
        # Clear frame references to free memory
        try:
            self.latest_frame = None
            self.last_face_annotations = []
            self.tracked_faces.clear()
            self.last_fire_detections = []
            self.last_weapon_detections = []
            self.last_vehicle_detections = []
            self.last_suspicious_detections = []
        except Exception:
            pass
    
    def __del__(self):
        """Clean up resources"""
        try:
            self.stop()
        except Exception:
            pass
    
    def _save_detection_log(self, frame, detected_name, bbox):
        """Save face detection to database with snapshot"""
        try:
            # Extract face region from frame
            top, right, bottom, left = bbox
            
            # Add some padding
            padding = 20
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(frame.shape[0], bottom + padding)
            right = min(frame.shape[1], right + padding)
            
            # Crop face region
            face_img = frame[top:bottom, left:right]
            
            # Encode as JPEG
            ret, jpeg = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ret:
                return
            
            # Create ContentFile for Django
            img_io = io.BytesIO(jpeg.tobytes())
            img_file = ContentFile(img_io.getvalue(), name=f'face_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.jpg')
            
            # Save to database
            DetectionLog.objects.create(
                event_name_id=self.event_id,
                camera_id=self.camera_id,
                detection_type='face',
                detected_label=detected_name,
                confidence_score=0.95,  # You can calculate actual confidence if available
                bounding_box={'top': int(top), 'right': int(right), 'bottom': int(bottom), 'left': int(left)},
                frame_snapshot=img_file
            )
        except Exception as e:
            print(f"Error saving detection log: {e}")
    
    def _save_fire_detection_log(self, frame, fire_detections):
        """Save fire detection to database with snapshot and create alert"""
        try:
            for detection in fire_detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                
                # Crop fire region with padding
                padding = 30
                y1_crop = max(0, y1 - padding)
                x1_crop = max(0, x1 - padding)
                y2_crop = min(frame.shape[0], y2 + padding)
                x2_crop = min(frame.shape[1], x2 + padding)
                
                fire_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                
                # Encode as JPEG
                ret, jpeg = cv2.imencode('.jpg', fire_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ret:
                    continue
                
                # Create ContentFile for Django
                img_io = io.BytesIO(jpeg.tobytes())
                img_file = ContentFile(img_io.getvalue(), name=f'fire_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.jpg')
                
                # Get camera name for alert
                try:
                    camera = RtspTable.objects.get(id=self.camera_id)
                    camera_name = camera.camera_name
                except:
                    camera_name = f"Camera {self.camera_id}"
                
                # Save to database
                log = DetectionLog.objects.create(
                    event_name_id=self.event_id,
                    camera_id=self.camera_id,
                    detection_type='fire',
                    detected_label=f'Fire detected at {camera_name}',
                    confidence_score=confidence,
                    bounding_box={'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                    frame_snapshot=img_file
                )
                
                print(f"🔥 FIRE ALERT: Fire detected at {camera_name} with confidence {confidence:.2f}")
                
        except Exception as e:
            print(f"Error saving fire detection log: {e}")
    
    def _save_weapon_detection_log(self, frame, weapon_detections):
        """Save weapon detection to database with snapshot and create alert"""
        try:
            for detection in weapon_detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                
                # Crop weapon region with padding
                padding = 30
                y1_crop = max(0, y1 - padding)
                x1_crop = max(0, x1 - padding)
                y2_crop = min(frame.shape[0], y2 + padding)
                x2_crop = min(frame.shape[1], x2 + padding)
                
                weapon_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                
                # Encode as JPEG
                ret, jpeg = cv2.imencode('.jpg', weapon_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ret:
                    continue
                
                # Create ContentFile for Django
                img_io = io.BytesIO(jpeg.tobytes())
                img_file = ContentFile(img_io.getvalue(), name=f'weapon_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.jpg')
                
                # Get camera name for alert
                try:
                    camera = RtspTable.objects.get(id=self.camera_id)
                    camera_name = camera.camera_name
                except:
                    camera_name = f"Camera {self.camera_id}"
                
                # Save to database
                log = DetectionLog.objects.create(
                    event_name_id=self.event_id,
                    camera_id=self.camera_id,
                    detection_type='weapon',
                    detected_label=f'Weapon detected at {camera_name}',
                    confidence_score=confidence,
                    bounding_box={'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                    frame_snapshot=img_file
                )
                
                print(f"⚠️ WEAPON ALERT: Weapon detected at {camera_name} with confidence {confidence:.2f}")
                
        except Exception as e:
            print(f"Error saving weapon detection log: {e}")
    
    def _save_vehicle_detection_log(self, frame, vehicle_detections):
        """Save vehicle detection to database with snapshot"""
        try:
            for detection in vehicle_detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                label = detection['label']
                
                # Crop vehicle region with padding
                padding = 20
                y1_crop = max(0, y1 - padding)
                x1_crop = max(0, x1 - padding)
                y2_crop = min(frame.shape[0], y2 + padding)
                x2_crop = min(frame.shape[1], x2 + padding)
                
                vehicle_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                
                # Encode as JPEG
                ret, jpeg = cv2.imencode('.jpg', vehicle_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue
                
                # Create ContentFile for Django
                img_io = io.BytesIO(jpeg.tobytes())
                img_file = ContentFile(img_io.getvalue(), name=f'vehicle_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.jpg')
                
                # Get camera name
                try:
                    camera = RtspTable.objects.get(id=self.camera_id)
                    camera_name = camera.camera_name
                except:
                    camera_name = f"Camera {self.camera_id}"
                
                # Save to database
                log = DetectionLog.objects.create(
                    event_name_id=self.event_id,
                    camera_id=self.camera_id,
                    detection_type='vehicle',
                    detected_label=f'{label} detected at {camera_name}',
                    confidence_score=confidence,
                    bounding_box={'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                    frame_snapshot=img_file
                )
                
        except Exception as e:
            print(f"Error saving vehicle detection log: {e}")
    
    def _save_suspicious_detection_log(self, frame, suspicious_detections):
        """Save suspicious activity detection to database with snapshot and create alert"""
        try:
            for detection in suspicious_detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                label = detection['label']
                
                # Crop suspicious activity region with padding
                padding = 25
                y1_crop = max(0, y1 - padding)
                x1_crop = max(0, x1 - padding)
                y2_crop = min(frame.shape[0], y2 + padding)
                x2_crop = min(frame.shape[1], x2 + padding)
                
                suspicious_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                
                # Encode as JPEG
                ret, jpeg = cv2.imencode('.jpg', suspicious_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue
                
                # Create ContentFile for Django
                img_io = io.BytesIO(jpeg.tobytes())
                img_file = ContentFile(img_io.getvalue(), name=f'suspicious_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.jpg')
                
                # Get camera name for alert
                try:
                    camera = RtspTable.objects.get(id=self.camera_id)
                    camera_name = camera.camera_name
                except:
                    camera_name = f"Camera {self.camera_id}"
                
                # Save to database
                log = DetectionLog.objects.create(
                    event_name_id=self.event_id,
                    camera_id=self.camera_id,
                    detection_type='suspicious',
                    detected_label=f'{label} detected at {camera_name}',
                    confidence_score=confidence,
                    bounding_box={'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                    frame_snapshot=img_file
                )
                
                print(f"🚨 SUSPICIOUS ALERT: {label} detected at {camera_name} with confidence {confidence:.2f}")
                
        except Exception as e:
            print(f"Error saving suspicious activity detection log: {e}")

    def _recognition_worker(self):
        """Background worker that performs face recognition on the latest frame.

        It downscales frames for performance and updates `self.last_face_annotations`.
        """
        try:
            while not self.stopped:
                try:
                    # Check if stopped
                    if self.stopped:
                        break
                    
                    # refresh flags periodically
                    self.refresh_ai_flags()

                    frame = None
                    try:
                        with self.lock:
                            if self.latest_frame is not None:
                                frame = self.latest_frame.copy()
                    except Exception:
                        frame = None

                    if frame is None:
                        # nothing to process yet
                        threading.Event().wait(0.05)
                        continue

                    # Downscale for performance
                    h, w = frame.shape[:2]
                    target_w = 320
                    scale = 1.0
                    if w > target_w:
                        scale = target_w / float(w)
                        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    else:
                        small = frame

                    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                    # Detect face locations on small image
                    locs_small = face_recognition.face_locations(rgb_small)

                    annotations = []
                    if self.face_recognition_mode:
                        # Only bounding boxes (no name matching)
                        for (top, right, bottom, left) in locs_small:
                            # map back to original coords
                            top_o = int(top / scale)
                            right_o = int(right / scale)
                            bottom_o = int(bottom / scale)
                            left_o = int(left / scale)
                            annotations.append(((top_o, right_o, bottom_o, left_o), None))
                    else:
                        # compute encodings on small image and match
                        encs_small = face_recognition.face_encodings(rgb_small, locs_small)
                        known = get_face_encodings_for_event(self.event_id)
                        known_names = [k[0] for k in known]
                        known_vecs = [k[1] for k in known]

                        for (top, right, bottom, left), enc in zip(locs_small, encs_small):
                            name = None
                            if len(known_vecs) > 0:
                                matches = face_recognition.compare_faces(known_vecs, enc, tolerance=0.45)
                                if True in matches:
                                    idx = matches.index(True)
                                    name = known_names[idx]

                            top_o = int(top / scale)
                            right_o = int(right / scale)
                            bottom_o = int(bottom / scale)
                            left_o = int(left / scale)
                            annotations.append(((top_o, right_o, bottom_o, left_o), name))

                    # update shared annotations
                    try:
                        with self.lock:
                            self.last_face_annotations = annotations
                    except Exception:
                        pass

                except Exception:
                    # resilience: don't break worker on errors
                    pass

                # small sleep to control CPU
                try:
                    threading.Event().wait(0.08)
                except Exception:
                    pass
        except Exception:
            # Final catch-all for thread safety
            pass

def run_onnx_inference(frame):
    input_blob = preprocess(frame)
    outputs = onnx_session.run([onnx_output], {onnx_input: input_blob})[0]
    return outputs

def draw_boxes(frame, preds):
    # preds = [x1, y1, x2, y2, score, class_id] per row
    for det in preds:
        x1, y1, x2, y2, score, cls = det
        if score < 0.50:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, str(cls), (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame


def model_processing(frame):

    # Convert JPEG bytes → image if needed
    np_frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)

    preds = run_onnx_inference(np_frame)
    
    # draw boxes
    processed = draw_boxes(np_frame, preds)

    # Encode back to JPEG
    ret, jpeg = cv2.imencode(".jpg", processed)
    return jpeg.tobytes()


def gen_frames(camera_id, enable_detection=True):
    """Generate frames for streaming with performance optimization"""
    frame_count = 0
    skip_frames = 0  # Dynamic frame skipping for web streaming
    
    if camera_id not in camera_streams:
        try:
            camera = RtspTable.objects.get(id=camera_id)
            video_camera = VideoCamera(
                camera.rtsp_url, 
                camera_id, 
                camera.event_name_id,
                enable_detection=enable_detection,
                stream_type=camera.stream_type
            )
            # Set initial face recognition mode based on AIModelEnabled
            cfg = AIModelEnabled.objects.filter(target_event_id=camera.event_name_id).first()
            if cfg:
                video_camera.face_recognition_mode = cfg.is_face_recognition_enabled
            camera_streams[camera_id] = video_camera
        except RtspTable.DoesNotExist:
            return
    
    camera = camera_streams[camera_id]
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        
        # Dynamic frame skipping based on system performance
        frame_count += 1
        
        # Check how many detection modes are active and adjust accordingly
        active_modes = sum([
            getattr(camera, 'fire_detection_mode', False),
            getattr(camera, 'weapon_detection_mode', False),
            getattr(camera, 'vehicle_detection_mode', False),
            getattr(camera, 'suspicious_detection_mode', False),
            getattr(camera, 'face_recognition_mode', False)
        ])
        
        # Skip frames for web streaming based on active detection load
        if active_modes > 2:
            skip_frames = 3  # Skip 2 out of every 3 frames
        elif active_modes > 1:
            skip_frames = 2  # Skip 1 out of every 2 frames
        else:
            skip_frames = 1  # No skipping
        
        if frame_count % skip_frames != 0:
            continue
        
        # model_processing handled separately; frame already annotated by VideoCamera
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request, camera_id):
    """Stream video feed from RTSP camera"""
    try:
        camera = get_object_or_404(RtspTable, id=camera_id)
        
        return StreamingHttpResponse(
            gen_frames(camera_id),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


def toggle_face_recognition(request, event_id):
    """API endpoint to toggle face recognition for an event.

    Expects POST with 'enabled'='true'|'false'. Updates AIModelEnabled and pushes to running streams.
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

    try:
        enabled = request.POST.get('enabled', 'false').lower() == 'true'

        # Ensure AIModelEnabled row exists for event
        cfg, _ = AIModelEnabled.objects.get_or_create(target_event_id=event_id)
        cfg.is_face_recognition_enabled = enabled
        cfg.save()

        # Push update to running camera streams for this event
        for cam_id, cam in camera_streams.items():
            try:
                if cam.event_id == event_id:
                    cam.face_recognition_mode = enabled
                    print(f"[toggle_face_recognition] Updated cam {cam_id} for event {event_id} -> face_recognition_mode={enabled}")
            except Exception:
                continue

        return JsonResponse({'status': 'success', 'enabled': enabled})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def get_face_recognition_status(request, event_id):
    """API endpoint to get current face recognition status for an event."""
    try:
        cfg = AIModelEnabled.objects.filter(target_event_id=event_id).first()
        if cfg:
            enabled = cfg.is_face_recognition_enabled
        else:
            enabled = False
        
        return JsonResponse({'status': 'success', 'enabled': enabled})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def toggle_fire_detection(request, event_id):
    """API endpoint to toggle fire detection for an event.
    
    Expects POST with 'enabled'='true'|'false'. Updates AIModelEnabled and pushes to running streams.
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)
    
    try:
        enabled = request.POST.get('enabled', 'false').lower() == 'true'
        
        # Ensure AIModelEnabled row exists for event
        cfg, _ = AIModelEnabled.objects.get_or_create(target_event_id=event_id)
        cfg.is_fire_detection_enabled = enabled
        cfg.save()
        
        # Push update to running camera streams for this event
        for cam_id, cam in camera_streams.items():
            try:
                if cam.event_id == event_id:
                    cam.fire_detection_mode = enabled
                    if not enabled:
                        cam.last_fire_detections = []  # Clear detections when disabled
                    print(f"[toggle_fire_detection] Updated cam {cam_id} for event {event_id} -> fire_detection_mode={enabled}")
            except Exception:
                continue
        
        return JsonResponse({'status': 'success', 'enabled': enabled})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def get_fire_detection_status(request, event_id):
    """API endpoint to get current fire detection status for an event."""
    try:
        cfg = AIModelEnabled.objects.filter(target_event_id=event_id).first()
        if cfg:
            enabled = cfg.is_fire_detection_enabled
        else:
            enabled = False
        
        return JsonResponse({'status': 'success', 'enabled': enabled})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def get_fire_alerts(request, event_id):
    """API endpoint to get recent fire detection alerts for an event."""
    try:
        # Get recent fire detections (last 50)
        fire_logs = DetectionLog.objects.filter(
            event_name_id=event_id,
            detection_type='fire'
        ).order_by('-created_at')[:50]
        
        alerts = []
        for log in fire_logs:
            try:
                camera = RtspTable.objects.get(id=log.camera_id)
                camera_name = camera.camera_name
            except:
                camera_name = f"Camera {log.camera_id}"
            
            alerts.append({
                'id': log.id,
                'camera_id': log.camera_id,
                'camera_name': camera_name,
                'detected_label': log.detected_label,
                'confidence': log.confidence_score,
                'snapshot_url': log.frame_snapshot.url if log.frame_snapshot else None,
                'timestamp': log.created_at.isoformat(),
                'bounding_box': log.bounding_box
            })
        
        return JsonResponse({
            'status': 'success',
            'alerts': alerts,
            'count': len(alerts)
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def toggle_weapon_detection(request, event_id):
    """API endpoint to toggle weapon detection for an event."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)
    
    try:
        enabled = request.POST.get('enabled', 'false').lower() == 'true'
        
        # Ensure AIModelEnabled row exists for event
        cfg, _ = AIModelEnabled.objects.get_or_create(target_event_id=event_id)
        cfg.is_weapon_detection_enabled = enabled
        cfg.save()
        
        # Push update to running camera streams for this event
        for cam_id, cam in camera_streams.items():
            try:
                if cam.event_id == event_id:
                    cam.weapon_detection_mode = enabled
                    if not enabled:
                        cam.last_weapon_detections = []  # Clear detections when disabled
                    print(f"[toggle_weapon_detection] Updated cam {cam_id} for event {event_id} -> weapon_detection_mode={enabled}")
            except Exception:
                continue
        
        return JsonResponse({'status': 'success', 'enabled': enabled})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def get_weapon_detection_status(request, event_id):
    """API endpoint to get current weapon detection status for an event."""
    try:
        cfg = AIModelEnabled.objects.filter(target_event_id=event_id).first()
        if cfg:
            enabled = cfg.is_weapon_detection_enabled
        else:
            enabled = False
        
        return JsonResponse({'status': 'success', 'enabled': enabled})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def toggle_vehicle_detection(request, event_id):
    """API endpoint to toggle vehicle detection for an event."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)
    
    try:
        enabled = request.POST.get('enabled', 'false').lower() == 'true'
        
        # Ensure AIModelEnabled row exists for event
        cfg, _ = AIModelEnabled.objects.get_or_create(target_event_id=event_id)
        cfg.is_vechicle_detection_enabled = enabled  # Note: field has typo in model
        cfg.save()
        
        # Push update to running camera streams for this event
        for cam_id, cam in camera_streams.items():
            try:
                if cam.event_id == event_id:
                    cam.vehicle_detection_mode = enabled
                    if not enabled:
                        cam.last_vehicle_detections = []  # Clear detections when disabled
                    print(f"[toggle_vehicle_detection] Updated cam {cam_id} for event {event_id} -> vehicle_detection_mode={enabled}")
            except Exception:
                continue
        
        return JsonResponse({'status': 'success', 'enabled': enabled})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def get_vehicle_detection_status(request, event_id):
    """API endpoint to get current vehicle detection status for an event."""
    try:
        cfg = AIModelEnabled.objects.filter(target_event_id=event_id).first()
        if cfg:
            enabled = cfg.is_vechicle_detection_enabled  # Note: field has typo in model
        else:
            enabled = False
        
        return JsonResponse({'status': 'success', 'enabled': enabled})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def toggle_suspicious_detection(request, event_id):
    """API endpoint to toggle suspicious activity detection for an event."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)
    
    try:
        enabled = request.POST.get('enabled', 'false').lower() == 'true'
        
        # Ensure AIModelEnabled row exists for event
        cfg, _ = AIModelEnabled.objects.get_or_create(target_event_id=event_id)
        cfg.is_suspicious_detection_enabled = enabled
        cfg.save()
        
        # Push update to running camera streams for this event
        for cam_id, cam in camera_streams.items():
            try:
                if cam.event_id == event_id:
                    cam.suspicious_detection_mode = enabled
                    if not enabled:
                        cam.last_suspicious_detections = []  # Clear detections when disabled
                    print(f"[toggle_suspicious_detection] Updated cam {cam_id} for event {event_id} -> suspicious_detection_mode={enabled}")
            except Exception:
                continue
        
        return JsonResponse({'status': 'success', 'enabled': enabled})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def get_suspicious_detection_status(request, event_id):
    """API endpoint to get current suspicious activity detection status for an event."""
    try:
        cfg = AIModelEnabled.objects.filter(target_event_id=event_id).first()
        if cfg:
            enabled = cfg.is_suspicious_detection_enabled
        else:
            enabled = False
        
        return JsonResponse({'status': 'success', 'enabled': enabled})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def get_threat_alerts(request, event_id):
    """API endpoint to get recent threat detection alerts (weapon, suspicious) for an event."""
    try:
        # Get recent threat detections (last 50)
        threat_logs = DetectionLog.objects.filter(
            event_name_id=event_id,
            detection_type__in=['weapon', 'suspicious']
        ).order_by('-created_at')[:50]
        
        alerts = []
        for log in threat_logs:
            try:
                camera = RtspTable.objects.get(id=log.camera_id)
                camera_name = camera.camera_name
            except:
                camera_name = f"Camera {log.camera_id}"
            
            alerts.append({
                'id': log.id,
                'camera_id': log.camera_id,
                'camera_name': camera_name,
                'detection_type': log.detection_type,
                'detected_label': log.detected_label,
                'confidence': log.confidence_score,
                'snapshot_url': log.frame_snapshot.url if log.frame_snapshot else None,
                'timestamp': log.created_at.isoformat(),
                'bounding_box': log.bounding_box
            })
        
        return JsonResponse({
            'status': 'success',
            'alerts': alerts,
            'count': len(alerts)
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def get_performance_stats(request, event_id):
    """API endpoint to get performance statistics for an event"""
    try:
        active_cameras = 0
        total_active_detections = 0
        performance_info = []
        
        for cam_id, cam in camera_streams.items():
            if cam.event_id == event_id:
                active_cameras += 1
                
                # Count active detection modes
                active_modes = {
                    'fire': getattr(cam, 'fire_detection_mode', False),
                    'weapon': getattr(cam, 'weapon_detection_mode', False),
                    'vehicle': getattr(cam, 'vehicle_detection_mode', False),
                    'suspicious': getattr(cam, 'suspicious_detection_mode', False),
                    'face': getattr(cam, 'face_recognition_mode', False)
                }
                
                active_count = sum(active_modes.values())
                total_active_detections += active_count
                
                performance_info.append({
                    'camera_id': cam_id,
                    'frame_count': getattr(cam, 'frame_count', 0),
                    'active_detections': active_count,
                    'skip_factor': getattr(cam, 'frame_skip_factor', 1),
                    'active_modes': active_modes
                })
        
        # Determine performance level
        if total_active_detections > 10:
            performance_level = 'heavy'
        elif total_active_detections > 5:
            performance_level = 'moderate' 
        else:
            performance_level = 'light'
        
        return JsonResponse({
            'status': 'success',
            'active_cameras': active_cameras,
            'total_active_detections': total_active_detections,
            'performance_level': performance_level,
            'cameras': performance_info
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


# ========================================
# FACE RECOGNITION VIEWS
# ========================================

def upload_face_model(request, event_id):
    """Upload a face recognition model"""
    if request.method == 'POST':
        try:
            event = get_object_or_404(Event, id=event_id)
            model_name = request.POST.get('model_name')
            model_file = request.FILES.get('model_file')
            
            if not model_file:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No model file provided'
                }, status=400)
            
            # Create face recognition model
            face_model = FaceRecognitionModel.objects.create(
                event_name=event,
                model_name=model_name,
                model_file=model_file,
                is_active=False
            )
            
            return JsonResponse({
                'status': 'success',
                'model_id': face_model.id,
                'message': 'Face recognition model uploaded successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def add_face_to_database(request, event_id):
    """Add a face to the recognition database"""
    if request.method == 'POST':
        try:
            event = get_object_or_404(Event, id=event_id)
            person_name = request.POST.get('person_name')
            face_image = request.FILES.get('face_image')
            
            if not face_image or not person_name:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Person name and face image required'
                }, status=400)
            
            # Create face database entry
            face_db = FaceDatabase.objects.create(
                event_name=event,
                person_name=person_name,
                face_image=face_image
            )

            # Invalidate cache for this event so new face is picked up
            try:
                face_encodings_cache.pop(event.id, None)
            except Exception:
                pass
            
            return JsonResponse({
                'status': 'success',
                'face_id': face_db.id,
                'message': f'Face for {person_name} added successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def get_face_models(request, event_id):
    """Get all face recognition models for an event"""
    try:
        event = get_object_or_404(Event, id=event_id)
        models = FaceRecognitionModel.objects.filter(event_name=event).values(
            'id', 'model_name', 'is_active', 'confidence_threshold', 'created_at'
        )
        
        return JsonResponse({
            'status': 'success',
            'models': list(models)
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


def get_face_database(request, event_id):
    """Get all registered faces for an event"""
    try:
        event = get_object_or_404(Event, id=event_id)
        faces = FaceDatabase.objects.filter(event_name=event)
        
        faces_data = []
        for face in faces:
            faces_data.append({
                'id': face.id,
                'person_name': face.person_name,
                'image_url': face.face_image.url if face.face_image else '',
                'created_at': face.created_at.isoformat()
            })
        
        return JsonResponse({
            'status': 'success',
            'faces': faces_data
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


def delete_face_model(request, model_id):
    """Delete a face recognition model"""
    if request.method == 'POST':
        try:
            model = get_object_or_404(FaceRecognitionModel, id=model_id)
            model.delete()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Face model deleted successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def delete_face_from_database(request, face_id):
    """Delete a face from the database"""
    if request.method == 'POST':
        try:
            face = get_object_or_404(FaceDatabase, id=face_id)
            face.delete()
            # Invalidate cache for this event to remove deleted face
            try:
                face_encodings_cache.pop(face.event_name_id, None)
            except Exception:
                pass
            return JsonResponse({
                'status': 'success',
                'message': 'Face deleted from database'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def activate_face_model(request, model_id):
    """Activate/Deactivate a face recognition model"""
    if request.method == 'POST':
        try:
            model = get_object_or_404(FaceRecognitionModel, id=model_id)
            is_active = request.POST.get('is_active', 'false').lower() == 'true'
            
            model.is_active = is_active
            model.save()
            
            return JsonResponse({
                'status': 'success',
                'message': f'Face model {model.model_name} {"activated" if is_active else "deactivated"}'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def detect_faces_in_image(request, event_id):
    """Detect faces in an uploaded image"""
    if request.method == 'POST':
        try:
            event = get_object_or_404(Event, id=event_id)
            image_file = request.FILES.get('image')
            
            if not image_file:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No image provided'
                }, status=400)
            
            # Read image
            nparr = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Simple face detection using Haar Cascade
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'confidence': 0.95
                })
            
            return JsonResponse({
                'status': 'success',
                'detections': detections,
                'count': len(detections)
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


# Object detection feature removed: related views deleted


def update_model_confidence(request, model_id):
    """Update confidence threshold for a face recognition model"""
    if request.method == 'POST':
        try:
            face_model = FaceRecognitionModel.objects.filter(id=model_id).first()
            confidence = float(request.POST.get('confidence', 0.5))

            if not face_model:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Face model not found'
                }, status=400)

            face_model.confidence_threshold = confidence
            face_model.save()

            return JsonResponse({
                'status': 'success',
                'message': 'Confidence threshold updated'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


# ========================================
# DETECTION LOGS VIEWS
# ========================================

def get_detection_logs(request, event_id):
    """API endpoint to get detection logs for an event"""
    try:
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 20))
        detection_type = request.GET.get('type', None)
        
        # Query logs
        logs = DetectionLog.objects.filter(event_name_id=event_id)
        
        if detection_type:
            logs = logs.filter(detection_type=detection_type)
        
        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        total_count = logs.count()
        logs = logs[start:end]
        
        # Serialize logs
        logs_data = []
        for log in logs:
            logs_data.append({
                'id': log.id,
                'camera_id': log.camera_id,
                'detection_type': log.detection_type,
                'detected_label': log.detected_label,
                'confidence_score': log.confidence_score,
                'bounding_box': log.bounding_box,
                'frame_snapshot': log.frame_snapshot.url if log.frame_snapshot else None,
                'created_at': log.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'time_ago': get_time_ago(log.created_at)
            })
        
        return JsonResponse({
            'status': 'success',
            'logs': logs_data,
            'total': total_count,
            'page': page,
            'page_size': page_size,
            'has_next': end < total_count
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


def delete_detection_log(request, log_id):
    """Delete a specific detection log"""
    if request.method == 'POST':
        try:
            log = DetectionLog.objects.get(id=log_id)
            
            # Delete the image file if exists
            if log.frame_snapshot:
                log.frame_snapshot.delete()
            
            log.delete()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Log deleted successfully'
            })
        except DetectionLog.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Log not found'
            }, status=404)
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def clear_detection_logs(request, event_id):
    """Clear all detection logs for an event"""
    if request.method == 'POST':
        try:
            logs = DetectionLog.objects.filter(event_name_id=event_id)
            count = logs.count()
            
            # Delete all images
            for log in logs:
                if log.frame_snapshot:
                    log.frame_snapshot.delete()
            
            logs.delete()
            
            return JsonResponse({
                'status': 'success',
                'message': f'Deleted {count} logs'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def generate_detection_report(request, event_id):
    """Generate a detection report for an event"""
    try:
        from django.db.models import Count
        from datetime import timedelta
        
        event = get_object_or_404(Event, id=event_id)
        
        # Get date range
        date_from = request.GET.get('date_from', None)
        date_to = request.GET.get('date_to', None)
        
        logs = DetectionLog.objects.filter(event_name_id=event_id)
        
        if date_from:
            logs = logs.filter(created_at__gte=date_from)
        if date_to:
            logs = logs.filter(created_at__lte=date_to)
        
        # Statistics
        total_detections = logs.count()
        
        # Group by detection type
        by_type = logs.values('detection_type').annotate(count=Count('id'))
        
        # Group by detected label
        by_label = logs.values('detected_label').annotate(count=Count('id')).order_by('-count')[:10]
        
        # Group by camera
        by_camera = logs.values('camera_id').annotate(count=Count('id'))
        
        # Recent detections
        recent = logs.order_by('-created_at')[:5]
        recent_data = []
        for log in recent:
            recent_data.append({
                'detected_label': log.detected_label,
                'detection_type': log.detection_type,
                'created_at': log.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return JsonResponse({
            'status': 'success',
            'report': {
                'event_name': event.event_name,
                'event_location': event.event_location,
                'total_detections': total_detections,
                'by_type': list(by_type),
                'by_label': list(by_label),
                'by_camera': list(by_camera),
                'recent_detections': recent_data
            }
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


def get_time_ago(dt):
    """Helper function to get human-readable time ago"""
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        return f"{int(seconds / 60)} minutes ago"
    elif seconds < 86400:
        return f"{int(seconds / 3600)} hours ago"
    else:
        return f"{int(seconds / 86400)} days ago"