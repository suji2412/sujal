from kivy.app import App
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
import cv2
import numpy as np

class ShapeDetectorApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.filechooser = FileChooserIconView()
        self.btn = Button(text='Detect Shapes')
        self.btn.bind(on_press=self.detect_shapes)
        self.image_widget = Image()
        
        self.layout.add_widget(self.filechooser)
        self.layout.add_widget(self.btn)
        self.layout.add_widget(self.image_widget)
        
        return self.layout

    def detect_shapes(self, instance):
        selected = self.filechooser.selection
        if selected:
            image_path = selected[0]
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                num_vertices = len(approx)
                if num_vertices == 3:
                    shape = "Triangle"
                elif num_vertices == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
                elif num_vertices == 5:
                    shape = "Pentagon"
                elif num_vertices > 5:
                    shape = "Circle"
                else:
                    shape = "Unknown"

                cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Save the modified image to display
            cv2.imwrite('output.png', image)
            self.image_widget.source = 'output.png'
            self.image_widget.reload()

if __name__ == '__main__':
    ShapeDetectorApp().run()
