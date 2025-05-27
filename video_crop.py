import cv2
import os

def extract_full_frames_at_intervals_cv2(video_path, output_folder, interval_seconds):
    """
    Extracts full frames from a video at defined intervals.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Folder to save the extracted frames.
        interval_seconds (float): The time interval between extracted frames in seconds.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
         print("Error: Could not get video FPS.")
         return

    # Calculate the frame interval based on seconds
    frame_interval = int(fps * interval_seconds)
    if frame_interval < 1:
         frame_interval = 1 # Ensure at least one frame per interval

    frame_count = 0
    saved_frame_count = 0

    print(f"Processing video with {fps} FPS. Saving full frames every {interval_seconds} seconds (approx. {frame_interval} frames).")

    while True:
        ret, frame = cap.read()

        if not ret:
            break # End of video

        # Check if the current frame is at the desired interval
        if frame_count % frame_interval == 0:
            # No cropping needed, just save the full frame
            output_filename = os.path.join(output_folder, f"full_frame_{saved_frame_count:04d}.png")

            # Save the full frame
            cv2.imwrite(output_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Finished processing. Saved {saved_frame_count} full frames to {output_folder}")

# --- Configuration ---
video_file = 'bladder.mp4' # Your original input video file (720p)
output_dir = 'full_frames_output' # Folder to save output images
save_interval = 0.5 # Save a frame every 0.5 seconds
# ---------------------

extract_full_frames_at_intervals_cv2(video_file, output_dir, save_interval)