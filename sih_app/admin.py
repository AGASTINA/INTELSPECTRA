from django.contrib import admin
from django.utils.html import format_html

from .models import (
	Event,
	RtspTable,
	AIModelEnabled,
	FaceRecognitionModel,
	FaceDatabase,
	DetectionLog,
)


class RtspTableInline(admin.TabularInline):
	model = RtspTable
	extra = 0
	fields = (
		'camera_name', 'stream_type', 'rtsp_url', 'camera_location', 'is_pinned', 'added_at'
	)
	readonly_fields = ('added_at',)


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
	list_display = ('event_name', 'event_type', 'event_location', 'event_created_date')
	search_fields = ('event_name', 'event_location', 'event_description')
	list_filter = ('event_type',)
	inlines = (RtspTableInline,)
	date_hierarchy = 'event_created_date'


@admin.register(RtspTable)
class RtspTableAdmin(admin.ModelAdmin):
	list_display = ('camera_name', 'event_name', 'stream_type', 'camera_location', 'is_pinned', 'added_at')
	search_fields = ('camera_name', 'rtsp_url', 'camera_location')
	list_filter = ('stream_type', 'is_pinned', 'event_name')
	readonly_fields = ('added_at',)
	list_per_page = 25


@admin.register(AIModelEnabled)
class AIModelEnabledAdmin(admin.ModelAdmin):
	list_display = (
		'target_event',
		'is_weapon_detection_enabled',
		'is_fire_detection_enabled',
		'is_face_covered_detection_enabled',
		'is_vechicle_detection_enabled',
		'is_suspicious_detection_enabled',
		'is_specific_object_detection_enabled',
		'is_face_recognition_enabled',
		'updated_at',
	)
	list_filter = (
		'is_weapon_detection_enabled', 'is_fire_detection_enabled', 'is_face_recognition_enabled', 'target_event'
	)
	search_fields = ('target_event__event_name',)
	readonly_fields = ('updated_at',)


@admin.register(FaceRecognitionModel)
class FaceRecognitionModelAdmin(admin.ModelAdmin):
	list_display = ('model_name', 'event_name', 'is_active', 'confidence_threshold', 'created_at', 'updated_at')
	search_fields = ('model_name', 'event_name__event_name')
	list_filter = ('is_active', 'event_name')
	readonly_fields = ('created_at', 'updated_at')
	actions = ['make_active', 'make_inactive']

	def make_active(self, request, queryset):
		updated = queryset.update(is_active=True)
		self.message_user(request, f"{updated} model(s) marked active")
	make_active.short_description = "Mark selected models as active"

	def make_inactive(self, request, queryset):
		updated = queryset.update(is_active=False)
		self.message_user(request, f"{updated} model(s) marked inactive")
	make_inactive.short_description = "Mark selected models as inactive"


@admin.register(FaceDatabase)
class FaceDatabaseAdmin(admin.ModelAdmin):
	list_display = ('person_name', 'event_name', 'created_at', 'face_image_tag')
	search_fields = ('person_name', 'event_name__event_name')
	list_filter = ('event_name',)
	readonly_fields = ('created_at',)

	def face_image_tag(self, obj):
		if obj.face_image and hasattr(obj.face_image, 'url'):
			return format_html('<img src="{}" style="max-height:80px;"/>', obj.face_image.url)
		return "-"
	face_image_tag.short_description = 'Face Image'


@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
	list_display = ('detection_type', 'detected_label', 'confidence_score', 'camera_id', 'event_name', 'created_at', 'frame_snapshot_tag')
	search_fields = ('detected_label', 'detection_type', 'event_name__event_name')
	list_filter = ('detection_type', 'event_name')
	readonly_fields = ('created_at',)
	date_hierarchy = 'created_at'

	def frame_snapshot_tag(self, obj):
		if obj.frame_snapshot and hasattr(obj.frame_snapshot, 'url'):
			return format_html('<img src="{}" style="max-height:120px;"/>', obj.frame_snapshot.url)
		return "-"
	frame_snapshot_tag.short_description = 'Snapshot'

