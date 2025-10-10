import subprocess
import time
import hashlib
import base64
import hmac
import json
import requests


# ========== 配置信息 ==========
APPID = "9a3c2b8c"
API_KEY = "5e9c2b9f42436e35636b07c2400a54de"
SECRET_KEY = "MDM0MmI3ODFjMDgwMTI4NjcyN2IyOGFm"

LFASR_HOST = "https://raasr.xfyun.cn/v2/api"


# ========== 第一步：提取音频 ==========
def extract_audio(video_path, audio_path):
    import os
    # Try to find FFmpeg in common locations
    ffmpeg_paths = [
        "ffmpeg",  # If it's in PATH
        os.path.expanduser("~\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.0-full_build\\bin\\ffmpeg.exe"),
        "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        "C:\\ffmpeg\\bin\\ffmpeg.exe"
    ]
    
    ffmpeg_cmd = None
    for path in ffmpeg_paths:
        if path == "ffmpeg":
            # Try to run ffmpeg directly (if in PATH)
            try:
                subprocess.run([path, "-version"], check=True, capture_output=True)
                ffmpeg_cmd = path
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        else:
            if os.path.exists(path):
                ffmpeg_cmd = path
                break
    
    if not ffmpeg_cmd:
        raise FileNotFoundError("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
    
    command = [
        ffmpeg_cmd,
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
        "-y"
    ]
    subprocess.run(command, check=True)
    print(f"Audio generated: {audio_path}")


# ========== 第二步：讯飞签名 ==========
def get_signa(ts):
    m2 = hashlib.md5()
    m2.update((APPID + str(ts)).encode('utf-8'))
    md5 = m2.hexdigest()
    baseString = (API_KEY + md5).encode('utf-8')
    signa = hmac.new(SECRET_KEY.encode('utf-8'), baseString, hashlib.sha1).digest()
    return base64.b64encode(signa).decode('utf-8')


# ========== 第三步：上传与获取结果 ==========
def lfasr_submit(file_path):
    ts = int(time.time())
    signa = get_signa(ts)
    files = {"file": open(file_path, "rb")}
    data = {
        "appId": APPID,
        "signa": signa,
        "ts": ts,
        "fileName": file_path,
        "fileSize": str(len(open(file_path, "rb").read()))
    }
    response = requests.post(f"{LFASR_HOST}/upload", data=data, files=files)
    return response.json()


def lfasr_get_result(task_id):
    ts = int(time.time())
    signa = get_signa(ts)
    data = {
        "appId": APPID,
        "signa": signa,
        "ts": ts,
        "taskId": task_id
    }
    response = requests.post(f"{LFASR_HOST}/getResult", data=data)
    return response.json()


# ========== 第四步：解析字幕 ==========
def parse_result(result_json, as_srt=False):
    """
    解析讯飞返回的转写结果
    as_srt=True 时返回 SRT 格式字幕
    """
    try:
        data = json.loads(result_json["data"])
    except Exception:
        return []

    subtitles = []
    index = 1
    for seg in data:
        onebest = seg.get("onebest", "")
        start_time = seg.get("bg", 0)  # 开始时间 (ms)
        end_time = seg.get("ed", 0)    # 结束时间 (ms)

        if as_srt:
            # 转换成 SRT 时间格式
            def ms_to_time(ms):
                h = ms // 3600000
                m = (ms % 3600000) // 60000
                s = (ms % 60000) // 1000
                ms_remain = ms % 1000
                return f"{h:02}:{m:02}:{s:02},{ms_remain:03}"

            subtitles.append(f"{index}\n{ms_to_time(start_time)} --> {ms_to_time(end_time)}\n{onebest}\n")
            index += 1
        else:
            subtitles.append(onebest)

    return subtitles


# ========== 主流程 ==========
def video_to_text(video_file, audio_file="output.wav", as_srt=False):
    extract_audio(video_file, audio_file)

    print("Uploading audio to iFlytek...")
    res = lfasr_submit(audio_file)
    print("Upload response:", res)
    task_id = res.get("data")

    if not task_id:
        print("Upload failed")
        return

    print("Waiting for recognition results...")
    while True:
        result = lfasr_get_result(task_id)
        if result["code"] == 0 and result["data"]:
            print("Transcription completed!")
            subtitles = parse_result(result, as_srt=as_srt)

            if as_srt:
                with open("output.srt", "w", encoding="utf-8") as f:
                    f.write("\n".join(subtitles))
                print("Subtitle file generated: output.srt")
            else:
                print("Text results:")
                print("\n".join(subtitles))
            break
        else:
            print("Still processing, retrying in 3 seconds...")
            time.sleep(3)


if __name__ == "__main__":
    # 纯文本输出
    video_to_text("1.mp4", "output.wav", as_srt=False)

    # 如果要 SRT 字幕，改成 as_srt=True
    # video_to_text("input.mp4", "output.wav", as_srt=True)
