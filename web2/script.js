document.addEventListener('DOMContentLoaded', function () {
    // Initialize canvas and UI elements
    const canvas = new fabric.Canvas('mainCanvas', { 
        selection: false,
        width: 800,
        height: 600,
        backgroundColor: '#f0f0f0'
    });

    // Get UI elements
    const pixelUmInput = document.getElementById('pixel_um');
    const depthUmInput = document.getElementById('depth_um');
    const fileUpload = document.getElementById('image_upload');
    const createRoiButton = document.getElementById('create_roi');
    const resetRoiButton = document.getElementById('reset_roi');
    const resetViewButton = document.getElementById('reset_view');
    const zoomInButton = document.getElementById('zoom_in');
    const zoomOutButton = document.getElementById('zoom_out');

    // State variables
    let baseImage = null;
    let zoomScale = 1;
    const ZOOM_IN_FACTOR = 1.125;
    const ZOOM_OUT_FACTOR = 0.875;
    
    // Helper function to fetch and update canvas data
    function updateCanvas(data) {
        // Clear canvas but keep the background
        canvas.clear();
        canvas.set('backgroundColor', '#f0f0f0');
        
        // Add back the base image if it exists
        if (baseImage) {
            canvas.add(baseImage);
        }

        // Draw points
        if (data.points) {
            data.points.forEach(point => {
                const circle = new fabric.Circle({
                    left: point.x - 5,
                    top: point.y - 5,
                    radius: 5,
                    fill: 'red',
                    stroke: 'white',
                    strokeWidth: 2,
                    selectable: false
                });
                canvas.add(circle);
            });
        }

        // Draw spline
        if (data.spline && data.spline.length > 1) {
            const points = data.spline.map(point => new fabric.Point(point.x, point.y));
            const spline = new fabric.Polyline(points, {
                stroke: 'blue',
                strokeWidth: 2,
                fill: null,
                selectable: false
            });
            canvas.add(spline);
        }

        // Draw ROI lines
        if (data.roi_lines) {
            data.roi_lines.forEach(line => {
                const lineObject = new fabric.Line(
                    [line[0].x, line[0].y, line[1].x, line[1].y],
                    {
                        stroke: 'green',
                        strokeWidth: 2,
                        selectable: false
                    }
                );
                canvas.add(lineObject);
            });
        }

        canvas.renderAll();
    }

    // Image upload handling
    function uploadImage(file) {
        const formData = new FormData();
        formData.append('image', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
            return fetch('/get_image');
        })
        .then(response => response.json())
        .then(data => {
            if (data.image) {
                fabric.Image.fromURL(
                    `data:image/png;base64,${data.image}`,
                    img => {
                        // Calculate scale to fit the canvas while maintaining aspect ratio
                        const scale = Math.min(
                            canvas.width / img.width,
                            canvas.height / img.height
                        );

                        img.scale(scale);
                        
                        // Center the image
                        img.center();
                        
                        // Set as base image and render
                        baseImage = img;
                        img.selectable = false;
                        
                        // Reset zoom and update canvas
                        zoomScale = 1;
                        canvas.setZoom(1);
                        updateCanvas({ points: [], spline: [], roi_lines: [] });
                    },
                    { crossOrigin: 'Anonymous' }
                );
            }
        })
        .catch(error => {
            console.error('Image upload error:', error);
            alert('Failed to upload image. Please try again.');
        });
    }

    // Update backend parameters
    function updateParameters() {
        const params = {
            pixel_um: parseFloat(pixelUmInput.value),
            depth_um: parseFloat(depthUmInput.value)
        };

        fetch('/update_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        })
        .catch(error => console.error('Parameter update error:', error));
    }

    // Fetch and update canvas data
    function fetchAndUpdateCanvas() {
        fetch('/get_data')
            .then(response => response.json())
            .then(data => updateCanvas(data))
            .catch(error => console.error('Data fetch error:', error));
    }

    // Event Handlers
    canvas.on('mouse:down', function(event) {
        const pointer = canvas.getPointer(event.e);
        fetch('/click', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                x: pointer.x,
                y: pointer.y
            })
        })
        .then(() => fetchAndUpdateCanvas())
        .catch(error => console.error('Click handling error:', error));
    });

    canvas.on('mouse:dblclick', function(event) {
        const pointer = canvas.getPointer(event.e);
        fetch('/remove_point', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                x: pointer.x,
                y: pointer.y
            })
        })
        .then(() => fetchAndUpdateCanvas())
        .catch(error => console.error('Point removal error:', error));
    });

    // Zoom functionality
    function zoom(factor) {
        zoomScale *= factor;
        canvas.setZoom(zoomScale);
        canvas.renderAll();
    }

    // UI Event Listeners
    fileUpload.addEventListener('change', function(e) {
        if (this.files && this.files[0]) {
            uploadImage(this.files[0]);
        }
    });

    createRoiButton.addEventListener('click', function() {
        updateParameters();
        fetch('/create_roi', { method: 'POST' })
            .then(() => fetchAndUpdateCanvas())
            .catch(error => console.error('ROI creation error:', error));
    });

    resetRoiButton.addEventListener('click', function() {
        fetch('/reset_roi', { method: 'POST' })
            .then(() => fetchAndUpdateCanvas())
            .catch(error => console.error('ROI reset error:', error));
    });

    resetViewButton.addEventListener('click', function() {
        if (baseImage) {
            zoomScale = 1;
            canvas.setZoom(1);
            baseImage.center();
            canvas.renderAll();
        }
    });

    zoomInButton.addEventListener('click', () => zoom(ZOOM_IN_FACTOR));
    zoomOutButton.addEventListener('click', () => zoom(ZOOM_OUT_FACTOR));

    // Canvas pan functionality
    let isDragging = false;
    let lastPosX;
    let lastPosY;

    canvas.on('mouse:down', function(opt) {
        const evt = opt.e;
        if (evt.altKey === true) {
            isDragging = true;
            canvas.selection = false;
            lastPosX = evt.clientX;
            lastPosY = evt.clientY;
        }
    });

    canvas.on('mouse:move', function(opt) {
        if (isDragging) {
            const evt = opt.e;
            const vpt = canvas.viewportTransform;
            vpt[4] += evt.clientX - lastPosX;
            vpt[5] += evt.clientY - lastPosY;
            canvas.requestRenderAll();
            lastPosX = evt.clientX;
            lastPosY = evt.clientY;
        }
    });

    canvas.on('mouse:up', function() {
        isDragging = false;
        canvas.selection = true;
    });

    // Initialize
    fetchAndUpdateCanvas();
});