import cv2
import numpy as np
import argparse

def extract_motion(video_path, output_path=None, offset_frames=1, 
                   threshold=15, blur_amount=0, background_color=(128, 128, 128)):

    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'avc1' for H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Failed to create video writer with avc1 codec.")
            print("Trying alternative codec (XVID)...")
            
            # Try XVID as fallback (requires .avi)
            avi_path = output_path.rsplit('.', 1)[0] + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print("Failed to create video writer. Please try a different output format.")
                return
    
    frame_buffer = []
    
    for _ in range(offset_frames + 1):
        ret, frame = cap.read()
        if not ret:
            print("Video too short for selected offset")
            return
        frame_buffer.append(frame)
    
    bg = np.ones((height, width, 3), dtype=np.uint8)
    bg[:,:] = background_color  # BGR format for OpenCV
    
    frame_count = offset_frames
    
    while True:
        # Get current frame and offset frame
        current_frame = frame_buffer[-1]
        offset_frame = frame_buffer[0]
        
        # Convert frames to grayscale for difference calculation
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        offset_gray = cv2.cvtColor(offset_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference between frames
        frame_diff = cv2.absdiff(current_gray, offset_gray)
        
        # Apply threshold to isolate moving parts
        _, motion_mask = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Optional: Apply blur to smooth the mask
        if blur_amount > 0:
            motion_mask = cv2.GaussianBlur(motion_mask, (blur_amount*2+1, blur_amount*2+1), 0)
            _, motion_mask = cv2.threshold(motion_mask, threshold//2, 255, cv2.THRESH_BINARY)
        
        output_frame = bg.copy()
        

        # Duplicate, invert, and make half transparent with time offset
        # This is just for the motion areas
        inverted_current = cv2.bitwise_not(current_frame)
        inverted_offset = cv2.bitwise_not(offset_frame)
        
        # Blend inverted frames with 50% opacity
        motion_visual = cv2.addWeighted(inverted_current, 0.5, inverted_offset, 0.5, 0)
        
        # Create mask and inverse mask (3 channels)
        mask_3ch = cv2.merge([motion_mask, motion_mask, motion_mask])
        inv_mask_3ch = cv2.bitwise_not(mask_3ch)
        
        moving_parts = cv2.bitwise_and(motion_visual, mask_3ch)
        
        # Extract static parts from the background
        static_parts = cv2.bitwise_and(output_frame, inv_mask_3ch)
        
        # Combine static background with moving parts
        output_frame = cv2.add(static_parts, moving_parts)
        
        frame_count += 1
        info_text = f"Frame: {frame_count}/{total_frames}, Offset: {offset_frames} frames"
        cv2.putText(output_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        cv2.imshow('Motion Extraction', output_frame)
        if output_path:
            out.write(output_frame)
        
        ret, next_frame = cap.read()
        if not ret:
            break
        
        frame_buffer.pop(0)
        frame_buffer.append(next_frame)
        
        #'q' to exit if  pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # free up resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Motion Extraction - Uniform gray background with colored motion')
    parser.add_argument('input', help='Path to the input video')
    parser.add_argument('--output', help='Path to the output video (optional)')
    parser.add_argument('--offset', type=int, default=1, 
                       help='Frame offset (1 for fast changes, higher for slower changes)')
    parser.add_argument('--threshold', type=int, default=15,
                       help='Threshold for motion detection (0-255)')
    parser.add_argument('--blur', type=int, default=0,
                       help='Blur amount for motion mask (0 for none)')
    parser.add_argument('--bg-color', type=str, default='128,128,128',
                       help='Background color for static parts as R,G,B (default: 128,128,128 for mid-gray)')
    
    args = parser.parse_args()
    
    try:
        bg_color = tuple(map(int, args.bg_color.split(',')))
        if len(bg_color) != 3:
            raise ValueError
        # Convert to BGR for OpenCV
        bg_color = (bg_color[2], bg_color[1], bg_color[0])
    except:
        print("Error: Background color must be specified as R,G,B (e.g., 128,128,128)")
        return
    
    extract_motion(
        args.input,
        args.output,
        args.offset,
        args.threshold,
        args.blur,
        bg_color
    )

if __name__ == "__main__":
    main()