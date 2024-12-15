import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.linear_model import LinearRegression
from PIL import Image, ImageDraw

class ImageProcessor:
    def __init__(self):
        self.points = []
        self.roi_lines = []
        self.current_image = None
    
    def set_image(self, image):
        """Initialize or update the current image"""
        if isinstance(image, np.ndarray):
            self.current_image = Image.fromarray(image)
        else:
            self.current_image = image
    
    def add_point(self, x: float, y: float) -> Image:
        """Add a point and return the updated image"""
        self.points.append((x, y))
        return self.draw_on_image()
    
    def reset_points(self) -> Image:
        """Reset all points and return clean image"""
        self.points = []
        return self.draw_on_image()
    
    def reset_roi(self) -> Image:
        """Reset ROI lines and return updated image"""
        self.roi_lines = []
        return self.draw_on_image()
    
    def create_bezier_spline(self, points):
        """Create a Bezier spline from given points"""
        if len(points) < 2:
            return []

        x = [p[0] for p in points]
        y = [p[1] for p in points]

        try:
            tck, _ = splprep([x, y], s=0)
            u = np.linspace(0, 1, num=500)
            spline_x, spline_y = splev(u, tck)
            return list(zip(spline_x, spline_y))
        except Exception as e:
            print(f"Spline creation error: {e}")
            return list(zip(x, y))

    def create_roi(self, pixel_um: float, depth_um: float) -> Image:
        """Create region of interest based on given parameters"""
        if len(self.points) < 2:
            return self.draw_on_image()

        try:
            depth_pixels = depth_um * pixel_um
            
            # Linear Regression
            x = np.array([p[0] for p in self.points]).reshape(-1, 1)
            y = np.array([p[1] for p in self.points])
            model = LinearRegression()
            model.fit(x, y)
            slope = model.coef_[0]
        
            # Calculate Unit Normal
            normal_x = -slope
            normal_y = 1
            norm = np.sqrt(normal_x**2 + normal_y**2)
            unit_normal = (normal_x / norm, normal_y / norm)
            
            # Shift First and Last Points
            first_point = self.points[0]
            last_point = self.points[-1]
            
            shifted_first = (
                first_point[0] + unit_normal[0] * depth_pixels,
                first_point[1] + unit_normal[1] * depth_pixels
            )
            shifted_last = (
                last_point[0] + unit_normal[0] * depth_pixels,
                last_point[1] + unit_normal[1] * depth_pixels
            )
            
            self.roi_lines = [
                (first_point, shifted_first),
                (shifted_first, shifted_last),
                (shifted_last, last_point)
            ]
            
        except Exception as e:
            print(f"Error creating ROI: {e}")
            
        return self.draw_on_image()

    def draw_on_image(self) -> Image:
        """Draw all points, splines and ROI on the image"""
        if self.current_image is None:
            return None
            
        img = self.current_image.copy()
        draw = ImageDraw.Draw(img)
        
        # Draw points
        for point in self.points:
            x, y = point
            draw.ellipse([x-5, y-5, x+5, y+5], fill="yellow")
            
        # Draw spline
        if len(self.points) >= 2:
            spline_points = self.create_bezier_spline(self.points)
            for i in range(len(spline_points) - 1):
                draw.line([spline_points[i], spline_points[i+1]], fill="green", width=2)
                
        # Draw ROI lines
        for line in self.roi_lines:
            draw.line([line[0], line[1]], fill="green", width=2)
            
        return img
