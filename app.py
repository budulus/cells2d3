import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QTransform, QCursor
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5 import uic
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.linear_model import LinearRegression

class BezierSplineTool(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load the UI file
        uic.loadUi('mainwindow.ui', self)
        
        # Initialize scene
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        
        # Connect signals
        self.load_button.clicked.connect(self.load_image)
        self.create_roi_button.clicked.connect(self.create_roi)
        self.reset_roi_button.clicked.connect(self.reset_roi)
        self.reset_view_button.clicked.connect(self.reset_view)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)

        # Initialize variables
        self.pixmap_item = None
        self.points = []
        self.drawing_enabled = False
        self.base_pixmap = None
        self.zoom_scale = 1.0
        self.scene_rect = QRectF()
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
            
            self.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            
            self.points = []
            self.drawing_enabled = True
            self.roi_lines = []
            self.update_display()

    def mousePressEvent(self, event):
        if not self.drawing_enabled or not self.base_pixmap:
            return
        
        if not self.edit_boundary_checkbox.isChecked():
            return

        if event.button() == Qt.LeftButton:
            cursor_pos = QCursor.pos()
            view_pos = self.graphicsView.mapFromGlobal(cursor_pos)
            scene_pos = self.graphicsView.mapToScene(view_pos)
            if self.scene.sceneRect().contains(scene_pos):
                self.points.append(scene_pos)
                self.update_display()
        elif event.button() == Qt.RightButton:
            cursor_pos = QCursor.pos()
            view_pos = self.graphicsView.mapFromGlobal(cursor_pos)
            scene_pos = self.graphicsView.mapToScene(view_pos)
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

            self.graphicsView.setTransform(QTransform().scale(self.zoom_scale, self.zoom_scale))

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

        # Draw ROI lines if they exist
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

        # Linear Regression
        x = np.array([p.x() for p in self.points]).reshape(-1, 1)
        y = np.array([p.y() for p in self.points])
        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_[0]
    
        # Calculate Unit Normal
        normal_x = -slope
        normal_y = 1
        norm = np.sqrt(normal_x**2 + normal_y**2)
        unit_normal = QPointF(normal_x / norm, normal_y / norm)
        
        # Shift First and Last Points
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
            self.graphicsView.setSceneRect(self.pixmap_item.boundingRect())
            self.graphicsView.fitInView(self.pixmap_item.boundingRect(), Qt.KeepAspectRatio)
            self.zoom_scale = 1.0
            self.graphicsView.setTransform(QTransform().scale(self.zoom_scale, self.zoom_scale))

    def zoom_in(self):
        self.zoom_scale *= 1.125
        self.graphicsView.setTransform(QTransform().scale(self.zoom_scale, self.zoom_scale))

    def zoom_out(self):
        self.zoom_scale *= 0.875
        self.graphicsView.setTransform(QTransform().scale(self.zoom_scale, self.zoom_scale))

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