from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [

    # Authentication
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    #home 
    path('', views.home, name='home'),

    #events management
    path('event/<int:event_id>/', views.event, name='event'),
    path('edit/<int:event_id>/', views.edit_event, name='edit_event'),
    path('delete/<int:event_id>/', views.delete_event, name='delete_event'),
    
    # Face Recognition page
    path('event/<int:event_id>/face-recognition/', views.face_recognition_page, name='face_recognition_page'),


    # Camera RTSP URLs
    path('camera/add/<int:event_id>/', views.add_rtsp_camera, name='add_camera'),
    path('camera/delete/<int:camera_id>/', views.delete_rtsp_camera, name='delete_camera'),
    path('camera/pin/<int:camera_id>/', views.pin_camera, name='pin_camera'),
    path('camera/unpin/<int:camera_id>/', views.unpin_camera, name='unpin_camera'),
    path('camera/list/<int:event_id>/', views.get_event_cameras, name='get_cameras'),
    path('camera/feed/<int:camera_id>/', views.video_feed, name='video_feed'),
    path('camera/test/', views.test_rtsp_connection, name='test_rtsp'),

    # Face Recognition URLs
    path('face/upload-model/<int:event_id>/', views.upload_face_model, name='upload_face_model'),
    path('face/add-to-database/<int:event_id>/', views.add_face_to_database, name='add_face_to_database'),
    path('face/models/<int:event_id>/', views.get_face_models, name='get_face_models'),
    path('face/database/<int:event_id>/', views.get_face_database, name='get_face_database'),
    path('face/delete-model/<int:model_id>/', views.delete_face_model, name='delete_face_model'),
    path('face/delete-from-database/<int:face_id>/', views.delete_face_from_database, name='delete_face_from_database'),
    path('face/activate-model/<int:model_id>/', views.activate_face_model, name='activate_face_model'),
    path('face/detect-image/<int:event_id>/', views.detect_faces_in_image, name='detect_faces_in_image'),

    # Object detection feature removed

    # Model Confidence URL
    path('model/update-confidence/<int:model_id>/', views.update_model_confidence, name='update_model_confidence'),
    # API to toggle face recognition for an event (updates running streams immediately)
    path('api/face-recognition/toggle/<int:event_id>/', views.toggle_face_recognition, name='toggle_face_recognition'),
    # API to get face recognition status for an event
    path('api/face-recognition/status/<int:event_id>/', views.get_face_recognition_status, name='get_face_recognition_status'),
    
    # Fire Detection URLs
    path('api/fire-detection/toggle/<int:event_id>/', views.toggle_fire_detection, name='toggle_fire_detection'),
    path('api/fire-detection/status/<int:event_id>/', views.get_fire_detection_status, name='get_fire_detection_status'),
    path('api/fire-detection/alerts/<int:event_id>/', views.get_fire_alerts, name='get_fire_alerts'),
    
    # Weapon Detection URLs
    path('api/weapon-detection/toggle/<int:event_id>/', views.toggle_weapon_detection, name='toggle_weapon_detection'),
    path('api/weapon-detection/status/<int:event_id>/', views.get_weapon_detection_status, name='get_weapon_detection_status'),
    
    # Vehicle Detection URLs
    path('api/vehicle-detection/toggle/<int:event_id>/', views.toggle_vehicle_detection, name='toggle_vehicle_detection'),
    path('api/vehicle-detection/status/<int:event_id>/', views.get_vehicle_detection_status, name='get_vehicle_detection_status'),
    
    # Suspicious Activity Detection URLs
    path('api/suspicious-detection/toggle/<int:event_id>/', views.toggle_suspicious_detection, name='toggle_suspicious_detection'),
    path('api/suspicious-detection/status/<int:event_id>/', views.get_suspicious_detection_status, name='get_suspicious_detection_status'),
    
    # Comprehensive Threat Alerts URL
    path('api/threat-alerts/<int:event_id>/', views.get_threat_alerts, name='get_threat_alerts'),
    
    # Performance Monitoring URL
    path('api/performance-stats/<int:event_id>/', views.get_performance_stats, name='get_performance_stats'),
    
    # Detection Logs URLs
    path('logs/<int:event_id>/', views.get_detection_logs, name='get_detection_logs'),
    path('logs/delete/<int:log_id>/', views.delete_detection_log, name='delete_detection_log'),
    path('logs/clear/<int:event_id>/', views.clear_detection_logs, name='clear_detection_logs'),
    path('logs/report/<int:event_id>/', views.generate_detection_report, name='generate_detection_report'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)