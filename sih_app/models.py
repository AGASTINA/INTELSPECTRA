from django.db import models
import os

class Event(models.Model):
    EVENT_TYPE_CHOICES = [
        ('Indoor', 'Indoor Operations'),
        ('Outdoor', 'Outdoor Operations'),
        ('Hybrid', 'Hybrid Operations'),
    ]
    
    event_name = models.CharField(max_length=200)
    event_description = models.TextField()
    event_location = models.CharField(max_length=200)
    event_type = models.CharField(max_length=20, choices=EVENT_TYPE_CHOICES)
    event_created_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.event_name
    
    class Meta:
        ordering = ['-event_created_date']
    
class RtspTable(models.Model):
    STREAM_TYPE_CHOICES = [
        ('rtsp', 'RTSP Stream'),
        ('rtmp', 'RTMP Stream'),
        ('http', 'HTTP/HTTPS Stream'),
        ('hls', 'HLS Stream (M3U8)'),
        ('usb', 'USB Camera'),
        ('file', 'Video File'),
        ('ip', 'IP Camera (HTTP MJPEG)'),
        ('other', 'Other Protocol'),
    ]
    
    event_name = models.ForeignKey(Event, on_delete=models.CASCADE)
    camera_name = models.CharField(max_length=100)
    rtsp_url = models.CharField(max_length=500, help_text="Stream URL or device index for USB camera")
    stream_type = models.CharField(max_length=20, choices=STREAM_TYPE_CHOICES, default='rtsp')
    camera_location = models.CharField(max_length=200, blank=True, null=True)
    is_pinned = models.BooleanField(default=False)
    added_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.camera_name} ({self.get_stream_type_display()})"
    
    class Meta:
        ordering = ['-added_at']


class AIModelEnabled(models.Model):
    target_event = models.ForeignKey(Event, on_delete=models.CASCADE)
    is_weapon_detection_enabled = models.BooleanField(default=False)
    is_fire_detection_enabled = models.BooleanField(default=False)
    is_face_covered_detection_enabled = models.BooleanField(default=False)
    is_vechicle_detection_enabled = models.BooleanField(default=False)
    is_suspicious_detection_enabled = models.BooleanField(default=False)
    is_specific_object_detection_enabled = models.BooleanField(default=False)
    is_face_recognition_enabled = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return "AI Model Enabled Settings"


class FaceRecognitionModel(models.Model):
    event_name = models.ForeignKey(Event, on_delete=models.CASCADE)
    model_name = models.CharField(max_length=200, unique=True)
    model_file = models.FileField(upload_to='face_recognition_models/')
    is_active = models.BooleanField(default=False)
    confidence_threshold = models.FloatField(default=0.6)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.model_name
    
    class Meta:
        ordering = ['-created_at']


class FaceDatabase(models.Model):
    event_name = models.ForeignKey(Event, on_delete=models.CASCADE)
    person_name = models.CharField(max_length=200)
    face_image = models.ImageField(upload_to='face_database/')
    face_embedding = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.person_name} - {self.event_name}"
    
    class Meta:
        ordering = ['-created_at']
class DetectionLog(models.Model):
    DETECTION_TYPES = [
        ('face', 'Face Recognition'),
        # object detection removed
        ('weapon', 'Weapon'),
        ('fire', 'Fire'),
        ('vehicle', 'Vehicle'),
    ]
    
    event_name = models.ForeignKey(Event, on_delete=models.CASCADE)
    camera_id = models.IntegerField()
    detection_type = models.CharField(max_length=50, choices=DETECTION_TYPES)
    detected_label = models.CharField(max_length=200)
    confidence_score = models.FloatField()
    bounding_box = models.JSONField(default=dict)
    frame_snapshot = models.ImageField(upload_to='detection_snapshots/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.detection_type} - {self.detected_label}"
    
    class Meta:
        ordering = ['-created_at']
