import gradio as gr
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.linear_model import LinearRegression
from PIL import Image, ImageDraw
import numpy as np

class BezierSplineTool:
    def __init__(self):
        self.points = []
        self.roi_lines = []
        self.current_image = None
        
    def reset_points(self):
        self.points = []
        return self.draw_on_image()
    
    def reset_roi(self):
        self.roi_lines = []
        return self.draw_on_image()
    
    def create_bezier_spline(self, points):
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

    def create_roi(self, pixel_um, depth_um):
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

    def draw_on_image(self):
        if self.current_image is None:
          return None
          
        # Convert the current image to a PIL image if it's a numpy array.
        if isinstance(self.current_image, np.ndarray):
            img = Image.fromarray(self.current_image)
        else:
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

    def add_point(self, image, evt: gr.SelectData):
      
        if image is not None:
          # Convert the image to a PIL image on initial load
          if self.current_image is None:
              if isinstance(image, np.ndarray):
                  self.current_image = Image.fromarray(image)
              else:
                  self.current_image = image

          self.points.append((evt.index[0], evt.index[1]))
          return self.draw_on_image()
        else:
          return None #Return None, so image doesn't disappear

def create_ui():
    tool = BezierSplineTool()
    
    with gr.Blocks() as app:
        gr.Markdown("# Bezier Spline Tool")
        
        with gr.Row():
            with gr.Column():
                # Updated to use interactive=True instead of tool="select"
                image_input = gr.Image(label="Input Image", interactive=True)
                
            with gr.Column():
                pixel_um = gr.Number(label="pixel/um", value=1)
                depth_um = gr.Number(label="depth [um]", value=10)
                
                with gr.Row():
                    create_roi_btn = gr.Button("Create ROI")
                    reset_roi_btn = gr.Button("Reset ROI")
                    reset_points_btn = gr.Button("Reset Points")
        
        # Event handlers
        image_input.select(
            fn=tool.add_point,
            inputs=[image_input],
            outputs=image_input
        )
        
        create_roi_btn.click(
            fn=tool.create_roi,
            inputs=[pixel_um, depth_um],
            outputs=image_input
        )
        
        reset_roi_btn.click(
            fn=tool.reset_roi,
            outputs=image_input
        )
        
        reset_points_btn.click(
            fn=tool.reset_points,
            outputs=image_input
        )
        
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True)
