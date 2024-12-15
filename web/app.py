import gradio as gr
from processor import ImageProcessor

def create_ui():
    processor = ImageProcessor()
    
    def handle_image_click(image, evt: gr.SelectData):
        if image is not None:
            if processor.current_image is None:
                processor.set_image(image)
            return processor.add_point(evt.index[0], evt.index[1])
        return None
    
    with gr.Blocks() as app:
        gr.Markdown("# Bezier Spline Tool")
        
        with gr.Row():
            with gr.Column():
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
            fn=handle_image_click,
            inputs=[image_input],
            outputs=image_input
        )
        
        create_roi_btn.click(
            fn=processor.create_roi,
            inputs=[pixel_um, depth_um],
            outputs=image_input
        )
        
        reset_roi_btn.click(
            fn=processor.reset_roi,
            outputs=image_input
        )
        
        reset_points_btn.click(
            fn=processor.reset_points,
            outputs=image_input
        )
        
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True)
