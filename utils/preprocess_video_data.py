import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tăng số lần thử đọc frame
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'


def process_videos(base_folder, frame_interval=15, diff_threshold=0.05, max_workers=4):
    video_folders = ['EasyLiveVideo', 'ImouVideo']
    image_folders = ['EasyLiveImage', 'ImouImage']

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for video_folder, image_folder in zip(video_folders, image_folders):
            video_path = os.path.join(base_folder, video_folder)
            image_path = os.path.join(base_folder, image_folder)

            if not os.path.exists(image_path):
                os.makedirs(image_path)

            for subdir in os.listdir(video_path):
                subdir_path = os.path.join(video_path, subdir)
                if os.path.isdir(subdir_path):
                    output_subdir = os.path.join(image_path, subdir)
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    for filename in os.listdir(subdir_path):
                        if filename.endswith((".mp4", ".avi", ".mov")):
                            video_file = os.path.join(subdir_path, filename)
                            executor.submit(process_video, video_file, output_subdir, frame_interval, diff_threshold)


def process_video(video_path, output_folder, frame_interval=15, diff_threshold=0.05):
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logging.error(f"Không thể mở video: {video_path}")
            return

        frame_count = 0
        saved_count = 0
        prev_frame = None

        video_prefix = datetime.now().strftime("%Y%m%d_%H%M%S") + "_"
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        while True:
            try:
                success, frame = video.read()
                if not success:
                    if frame_count == 0:
                        logging.error(f"Không thể đọc bất kỳ frame nào từ video: {video_path}")
                    break

                if frame_count % frame_interval == 0:
                    if prev_frame is None:
                        save_frame(frame, output_folder, video_prefix, video_name, saved_count)
                        saved_count += 1
                        prev_frame = frame
                    else:
                        diff = frame_difference(prev_frame, frame)
                        logging.debug(f"Frame {frame_count}: Difference = {diff:.4f}")
                        if diff > diff_threshold:
                            save_frame(frame, output_folder, video_prefix, video_name, saved_count)
                            saved_count += 1
                            prev_frame = frame
                        else:
                            logging.debug(f"Frame {frame_count}: Skipped (below threshold)")

                frame_count += 1

            except cv2.error as e:
                logging.warning(f"Lỗi khi đọc frame {frame_count} từ {video_path}: {str(e)}")
                frame_count += 1
                continue

        video.release()
        logging.info(f"Xử lý xong {video_path}. {saved_count}/{frame_count} frames đã lưu.")
    except Exception as e:
        logging.error(f"Lỗi khi xử lý video {video_path}: {str(e)}")


def frame_difference(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return np.mean(diff) / 255.0


def save_frame(frame, output_folder, video_prefix, video_name, count):
    filename = os.path.join(output_folder, f"{video_prefix}{video_name}_frame_{count:04d}.jpg")
    try:
        cv2.imwrite(filename, frame)
    except Exception as e:
        logging.error(f"Không thể lưu frame {count}: {str(e)}")


if __name__ == "__main__":
    base_folder = r"C:\Users\Administrator\Desktop\video_data"
    process_videos(base_folder, frame_interval=15, diff_threshold=0.05, max_workers=4)