from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
import os
import shutil
import threading


VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv')


def concat_videos(folder: str, output_path: str) -> str:
    """
    Concatenates all video files in a folder into one, sorted by filename.
    After completion, schedules the folder for deletion after 2 days.

    Args:
        folder: Path to the folder containing video files (same format).
        output_path: Path for the final concatenated video file.

    Returns:
        The output file path.
    """
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )

    if len(files) < 2:
        raise ValueError(f"Need at least 2 video files in folder, found {len(files)}")

    clips = []
    try:
        for f in files:
            clips.append(VideoFileClip(os.path.join(folder, f)))

        final = concatenate_videoclips(clips)
        final.write_videofile(output_path, logger=None)
        final.close()
    finally:
        for clip in clips:
            clip.close()

    _schedule_folder_deletion(folder, delay_seconds=2 * 24 * 60 * 60)

    return output_path


def _schedule_folder_deletion(folder: str, delay_seconds: float):
    """Deletes a folder after a delay using a background timer."""
    def _delete():
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)

    timer = threading.Timer(delay_seconds, _delete)
    timer.daemon = True
    timer.start()


def merge_video_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Combines a video file with an audio file. The final video duration
    matches the audio duration — the video is trimmed or looped as needed.

    Args:
        video_path: Path to the video file.
        audio_path: Path to the audio file.
        output_path: Path for the output file.

    Returns:
        The output file path.
    """
    audio = AudioFileClip(audio_path)
    video = VideoFileClip(video_path)

    try:
        video = video.with_duration(audio.duration)
        video = video.with_audio(audio)
        video.write_videofile(output_path, logger=None)
    finally:
        video.close()
        audio.close()

    return output_path
