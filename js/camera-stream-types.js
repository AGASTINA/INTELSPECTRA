// Updated JavaScript snippet for event.html - Add Camera Modal Enhancement
// Replace the existing Add Camera Modal form in event.html with this enhanced version

/*
UPDATED ADD CAMERA MODAL HTML:
Replace the modal body in event.html with:

<div class="modal-body">
    <form id="addCameraForm">
        <div class="mb-3">
            <label class="form-label">Camera Name</label>
            <input type="text" class="form-control" id="cameraName" required>
        </div>
        
        <div class="mb-3">
            <label class="form-label">Stream Type</label>
            <select class="form-select" id="streamType" required>
                <option value="rtsp">RTSP Stream</option>
                <option value="rtmp">RTMP Stream</option>
                <option value="http">HTTP/HTTPS Stream</option>
                <option value="hls">HLS Stream (M3U8)</option>
                <option value="ip">IP Camera (MJPEG)</option>
                <option value="usb">USB Camera</option>
                <option value="file">Video File</option>
                <option value="other">Other Protocol</option>
            </select>
            <small class="form-text text-muted" id="streamTypeHelp">
                Select the type of video source
            </small>
        </div>
        
        <div class="mb-3">
            <label class="form-label" id="urlLabel">Stream URL / Device</label>
            <input type="text" class="form-control" id="rtspUrl" required>
            <small class="form-text text-muted" id="urlHelp">
                Enter the stream URL or device index
            </small>
        </div>
        
        <div class="mb-3">
            <label class="form-label">Camera Location (Optional)</label>
            <input type="text" class="form-control" id="cameraLocation">
        </div>
        
        <button type="button" class="btn btn-secondary" onclick="testStreamConnection()">
            Test Connection
        </button>
    </form>
</div>
*/

// Add this JavaScript to update the form based on stream type selection

document.addEventListener('DOMContentLoaded', function() {
    const streamTypeSelect = document.getElementById('streamType');
    const urlInput = document.getElementById('rtspUrl');
    const urlLabel = document.getElementById('urlLabel');
    const urlHelp = document.getElementById('urlHelp');
    
    if (streamTypeSelect) {
        streamTypeSelect.addEventListener('change', function() {
            const streamType = this.value;
            
            // Update labels and placeholders based on stream type
            switch(streamType) {
                case 'rtsp':
                    urlLabel.textContent = 'RTSP URL';
                    urlInput.placeholder = 'rtsp://username:password@ip:port/path';
                    urlHelp.textContent = 'Example: rtsp://admin:pass@192.168.1.100:554/stream1';
                    break;
                    
                case 'rtmp':
                    urlLabel.textContent = 'RTMP URL';
                    urlInput.placeholder = 'rtmp://server/live/stream';
                    urlHelp.textContent = 'Example: rtmp://192.168.1.100:1935/live/camera1';
                    break;
                    
                case 'http':
                    urlLabel.textContent = 'HTTP/HTTPS URL';
                    urlInput.placeholder = 'http://camera-ip/video';
                    urlHelp.textContent = 'Example: http://192.168.1.100:8080/video';
                    break;
                    
                case 'hls':
                    urlLabel.textContent = 'HLS URL (M3U8)';
                    urlInput.placeholder = 'http://server/stream/playlist.m3u8';
                    urlHelp.textContent = 'Example: https://cdn.example.com/live/stream.m3u8';
                    break;
                    
                case 'ip':
                    urlLabel.textContent = 'IP Camera URL (MJPEG)';
                    urlInput.placeholder = 'http://ip/video.mjpg';
                    urlHelp.textContent = 'Example: http://192.168.1.100/video.mjpeg';
                    break;
                    
                case 'usb':
                    urlLabel.textContent = 'USB Device Index';
                    urlInput.placeholder = '0';
                    urlHelp.textContent = 'Enter device index: 0 for first camera, 1 for second, etc.';
                    urlInput.value = '0'; // Default to first USB camera
                    break;
                    
                case 'file':
                    urlLabel.textContent = 'Video File Path';
                    urlInput.placeholder = '/path/to/video/file.mp4';
                    urlHelp.textContent = 'Example: /media/recordings/footage.mp4';
                    break;
                    
                case 'other':
                    urlLabel.textContent = 'Stream URL / Connection String';
                    urlInput.placeholder = 'Enter connection string';
                    urlHelp.textContent = 'Enter the appropriate connection string for your protocol';
                    break;
            }
        });
    }
});

// Updated test connection function
function testStreamConnection() {
    const streamType = document.getElementById('streamType').value;
    const rtspUrl = document.getElementById('rtspUrl').value.trim();
    
    if (!rtspUrl) {
        alert('Please enter a stream URL or device index');
        return;
    }
    
    // Show loading state
    const testBtn = event.target;
    const originalText = testBtn.textContent;
    testBtn.disabled = true;
    testBtn.textContent = 'Testing...';
    
    fetch('/camera/test/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: `rtsp_url=${encodeURIComponent(rtspUrl)}&stream_type=${encodeURIComponent(streamType)}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert('✓ ' + data.message);
        } else {
            alert('✗ Connection failed: ' + data.message);
        }
    })
    .catch(error => {
        alert('✗ Error testing connection: ' + error);
    })
    .finally(() => {
        testBtn.disabled = false;
        testBtn.textContent = originalText;
    });
}

// Updated add camera function
function addCameraToEvent() {
    const cameraName = document.getElementById('cameraName').value.trim();
    const streamType = document.getElementById('streamType').value;
    const rtspUrl = document.getElementById('rtspUrl').value.trim();
    const cameraLocation = document.getElementById('cameraLocation').value.trim();
    
    if (!cameraName || !rtspUrl) {
        alert('Please fill in required fields');
        return;
    }
    
    const eventId = document.body.getAttribute('data-event-id');
    
    fetch(`/camera/add/${eventId}/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: `camera_name=${encodeURIComponent(cameraName)}&stream_type=${encodeURIComponent(streamType)}&rtsp_url=${encodeURIComponent(rtspUrl)}&camera_location=${encodeURIComponent(cameraLocation)}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert('✓ Camera added successfully!');
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('addCameraModal'));
            modal.hide();
            
            // Reset form
            document.getElementById('addCameraForm').reset();
            
            // Reload cameras
            loadCameras();
        } else {
            alert('✗ Error: ' + data.message);
        }
    })
    .catch(error => {
        alert('✗ Error adding camera: ' + error);
    });
}

// Helper function to get CSRF token
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Display stream type badge in camera cards
function getStreamTypeBadge(streamType) {
    const badges = {
        'rtsp': '<span class="badge bg-primary">RTSP</span>',
        'rtmp': '<span class="badge bg-info">RTMP</span>',
        'http': '<span class="badge bg-success">HTTP</span>',
        'hls': '<span class="badge bg-warning">HLS</span>',
        'ip': '<span class="badge bg-secondary">IP CAM</span>',
        'usb': '<span class="badge bg-dark">USB</span>',
        'file': '<span class="badge bg-light text-dark">FILE</span>',
        'other': '<span class="badge bg-secondary">OTHER</span>'
    };
    
    return badges[streamType] || '<span class="badge bg-secondary">UNKNOWN</span>';
}
