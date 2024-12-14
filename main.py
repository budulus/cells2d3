import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QFileDialog, QWidget,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QCheckBox, QLineEdit, QLabel, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QTransform, QCursor
from PyQt5.QtCore import Qt, QPointF, QRectF
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.linear_model import LinearRegression

class BezierSplineTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Bezier Spline Tool")
        
        # Main widget and layout
        self.main_widget = QWidget()
        self.layout = QVBoxLayout(self.main_widget)

        # Graphics View and Scene Setup
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.layout.addWidget(self.view)

        # Input Layout
        input_layout = QHBoxLayout()

        # pixel/um input
        self.pixel_um_label = QLabel("pixel/um:")
        input_layout.addWidget(self.pixel_um_label)
        self.pixel_um_input = QLineEdit("1") # default 1
        input_layout.addWidget(self.pixel_um_input)

        # depth [um] input
        self.depth_label = QLabel("depth [um]:")
        input_layout.addWidget(self.depth_label)
        self.depth_input = QLineEdit("10") # default 10
        input_layout.addWidget(self.depth_input)

        self.layout.addLayout(input_layout)

        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        # Edit boundary checkbox
        self.edit_boundary_checkbox = QCheckBox("Edit Boundary")
        self.edit_boundary_checkbox.setChecked(True)  # Default to checked
        self.layout.addWidget(self.edit_boundary_checkbox)

        # Create ROI button
        self.create_roi_button = QPushButton("Create ROI")
        self.create_roi_button.clicked.connect(self.create_roi)
        self.layout.addWidget(self.create_roi_button)

        # Reset ROI button
        self.reset_roi_button = QPushButton("Reset ROI")
        self.reset_roi_button.clicked.connect(self.reset_roi)
        self.layout.addWidget(self.reset_roi_button)

        # Reset View button
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_view)
        self.layout.addWidget(self.reset_view_button)

        
        # Zoom buttons
        zoom_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.zoom_out_button)
        self.layout.addLayout(zoom_layout)


        self.setCentralWidget(self.main_widget)

        self.setCentralWidget(self.main_widget)

        # Variables to store image and points
        self.pixmap_item = None
        self.points = []
        self.drawing_enabled = False
        self.base_pixmap = None
        self.zoom_scale = 1.0  # Initial zoom scale
        self.scene_rect = QRectF()  # Initialize scene_rect
        self.roi_lines = []

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            self.pixmap = QPixmap(file_name)
            self.base_pixmap = self.pixmap.copy()

            if self.pixmap_item:
                self.scene.removeItem(self.pixmap_item)

            self.pixmap_item = QGraphicsPixmapItem(self.base_pixmap)
            self.scene.addItem(self.pixmap_item)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio) # fit the view to the scene
            
            self.points = []
            self.drawing_enabled = True
            self.roi_lines = []
            self.update_display()

    def mousePressEvent(self, event):
        if not self.drawing_enabled or not self.base_pixmap:
            return
        
        if not self.edit_boundary_checkbox.isChecked():
             return #Do nothing if edit boundary is not checked

        if event.button() == Qt.LeftButton:
            # Use the cursor's hot spot to get the correct click position
            cursor_pos = QCursor.pos()
            view_pos = self.view.mapFromGlobal(cursor_pos)
            scene_pos = self.view.mapToScene(view_pos)
            if self.scene.sceneRect().contains(scene_pos):
                self.points.append(scene_pos)
                self.update_display()
        elif event.button() == Qt.RightButton:
            # Use the cursor's hot spot to get the correct click position
            cursor_pos = QCursor.pos()
            view_pos = self.view.mapFromGlobal(cursor_pos)
            scene_pos = self.view.mapToScene(view_pos)
            if self.points:
                self.points = [p for p in self.points if (p - scene_pos).manhattanLength() > 10]
                self.update_display()

    def wheelEvent(self, event):
         if self.drawing_enabled and self.base_pixmap:
            zoom_factor = 1.1
            if event.angleDelta().y() > 0:
                self.zoom_scale *= zoom_factor
            else:
                self.zoom_scale /= zoom_factor

            self.view.setTransform(QTransform().scale(self.zoom_scale, self.zoom_scale))

    def update_display(self):
        if not self.base_pixmap:
            return

        if self.pixmap_item:
          self.scene.removeItem(self.pixmap_item)
        
        pixmap = QPixmap(self.base_pixmap)
        painter = QPainter(pixmap)

        # Draw points
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("yellow"))
        for point in self.points:
            painter.drawEllipse(point, 5, 5)

        # Draw spline
        if len(self.points) >= 2:
             try:
                spline_points = self.create_bezier_spline(self.points)
                pen = QPen(QColor("green"), 2)
                painter.setPen(pen)
                path = [spline_points[0]]
                for point in spline_points[1:]:
                    painter.drawLine(path[-1], point)
                    path.append(point)
             except Exception as e:
                print(f"Error creating spline: {e}")

        #Draw ROI lines if they exist
        if self.roi_lines:
          pen = QPen(QColor("green"), 2)
          painter.setPen(pen)
          for line in self.roi_lines:
            painter.drawLine(line[0], line[1])

        painter.end()

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
    

    def create_roi(self):
        if len(self.points) < 2:
            return

        try:
            pixel_um = float(self.pixel_um_input.text())
            depth_um = float(self.depth_input.text())
        except ValueError:
            print("Invalid input for pixel/um or depth [um]")
            return
        
        depth_pixels = depth_um * pixel_um

        # 1. Linear Regression
        x = np.array([p.x() for p in self.points]).reshape(-1, 1)
        y = np.array([p.y() for p in self.points])
        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_[0]
    
        # 2. Calculate Unit Normal
        normal_x = -slope
        normal_y = 1
        norm = np.sqrt(normal_x**2 + normal_y**2)
        unit_normal = QPointF(normal_x / norm, normal_y / norm)
        
        # 3. Shift First and Last Points
        first_point = self.points[0]
        last_point = self.points[-1]
        
        shifted_first_point = first_point + (unit_normal * depth_pixels)
        shifted_last_point = last_point + (unit_normal * depth_pixels)
        
        # Create the line segments for the ROI
        self.roi_lines = [
            (first_point, shifted_first_point),
            (shifted_first_point, shifted_last_point),
            (shifted_last_point, last_point)
        ]
        self.update_display()

    def reset_roi(self):
        self.roi_lines = []
        self.update_display()
    
    def reset_view(self):
        if self.pixmap_item:
           self.view.setSceneRect(self.pixmap_item.boundingRect())
           self.view.fitInView(self.pixmap_item.boundingRect(), Qt.KeepAspectRatio)
           self.zoom_scale = 1.0
           self.view.setTransform(QTransform().scale(self.zoom_scale, self.zoom_scale))

    def zoom_in(self):
        self.zoom_scale *= 1.125  # Increase by 12.5%
        self.view.setTransform(QTransform().scale(self.zoom_scale, self.zoom_scale))

    def zoom_out(self):
        self.zoom_scale *= 0.875 # Decrease by 12.5% (1 - 0.125 = 0.875)
        self.view.setTransform(QTransform().scale(self.zoom_scale, self.zoom_scale))

    def create_bezier_spline(self, points):
        if len(points) < 2:
            return []

        x = [p.x() for p in points]
        y = [p.y() for p in points]

        try:
            tck, _ = splprep([x, y], s=0)
            u = np.linspace(0, 1, num=500)
            spline_x, spline_y = splev(u, tck)
            return [QPointF(px, py) for px, py in zip(spline_x, spline_y)]
        except Exception as e:
            print(f"Spline creation error: {e}")
            return [QPointF(px, py) for px, py in zip(x, y)]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BezierSplineTool()
    window.show()
    sys.exit(app.exec_())
