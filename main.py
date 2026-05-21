from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaFileUpload
from concurrent.futures import ThreadPoolExecutor
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
from time import sleep
import configparser
import subprocess
import threading
import whisper
import pickle
import socket
import torch
import queue
import re
import os

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

config = configparser.ConfigParser()
config.read("conf.ini")

TARGET_DIRECTORY = Path(config["Paths"]["target_directory"]).resolve(strict=True)
APPLICATION_DATA_DIRECTORY = Path(config["Paths"]["application_data_directory"])
CREDENTIALS_FILE = Path(config["Paths"]["credentials_file"]).resolve()
EXTENSIONS = config["Conf"]["extensions"].split(",")
WORD_BLACKLIST = config["Conf"]["word_blacklist"].split(",")

API_KEY = config["YouTube"]["api_key"]
CLIENT_SECRETS_FILE = config["YouTube"]["client_secrets_file"]

# Configuration
MAX_TRANSCRIBE_WORKERS = 4
MAX_UPLOAD_WORKERS = 1

# Control flags - set these to enable features
ALLOW_YOUTUBE_UPLOADS = True
ALLOW_TRANSCRIBING = True
FORCE_RESET_BLACKLIST = False
FORCE_RESET_YOUTUBE = False

_upload_lock = []
_transcribe_lock = []

metadata_files = []

exit_event = threading.Event()

# Queues for managing work
transcribe_queue = queue.Queue()
upload_queue = queue.Queue()

# Thread pools
transcribe_executor = ThreadPoolExecutor(max_workers=MAX_TRANSCRIBE_WORKERS, thread_name_prefix="Transcribe")
upload_executor = ThreadPoolExecutor(max_workers=MAX_UPLOAD_WORKERS, thread_name_prefix="Upload")


class MetadataFile:
    def __init__(self, model, vid_filepath: Path):
        """Makes a new MetadataFile class"""
        self.filepath = Path(APPLICATION_DATA_DIRECTORY / vid_filepath.with_suffix(".ysd").name)
        
        self.vid_filepath = vid_filepath
        self.transcript = ""
        self.blacklist_status = False
        self.youtube_uploaded = False
        self.youtube_link = ""
        self.resumable_uri = ""
        self.resumable_upload_progress = 0
        
        self.model = model
        
        load = self.from_serialized(self.filepath)
        if load is not None:
            self.vid_filepath = load.vid_filepath
            self.transcript = load.transcript or ""
            self.blacklist_status = load.blacklist_status or False
            
            try:
                self.youtube_uploaded = load.youtube_uploaded
                self.youtube_link = load.youtube_link
                
                if load.youtube_link:
                    self.resumable_uri = "UPLOADED"
                    self.resumable_upload_progress = -1
                    
            except AttributeError:
                log("[?] File is missing some new-version fields, they will be added.")
                self.youtube_uploaded = False
                self.youtube_link = ""
                self.resumable_uri = ""
                self.resumable_upload_progress = 0
        
        if FORCE_RESET_BLACKLIST:
            self.check_blacklisted()
        
        if FORCE_RESET_YOUTUBE:
            self.youtube_uploaded = False
            self.youtube_link = ""
    
        
    def transcribe_to_text(self):
        # Skip if already transcribed
        if self.transcript != "":
            log(f"[~] Transcript already exists for {self.vid_filepath.name}, skipping")
            return
        
        # Validate file first
        is_valid, msg = validate_video_file(self.vid_filepath)
        if not is_valid:
            log(f"[!] Skipping {self.vid_filepath.name}: {msg}")
            self.transcript = "INVALID_FILE"
            self.to_serialized()
            return
        
        lock_name = self.vid_filepath.stem
        _transcribe_lock.append(lock_name)
        
        audio_path = None
        try:
            audio_path = extract_audio_ffmpeg(self.vid_filepath, self.vid_filepath.with_suffix(".mp3"))
            log(f"[*] Stripped audio to {audio_path}")
            
            # Verify extracted audio isn't empty
            if audio_path.stat().st_size < 1024:  # Less than 1KB
                log(f"[!] Extracted audio is too small ({audio_path.stat().st_size} bytes), skipping")
                self.transcript = "INVALID_AUDIO"
                self.to_serialized()
                return
            
            try:
                result = self.model.transcribe(str(audio_path), verbose=False)
                self.transcript = result["text"] if result.get("text") not in ("", " ", None) else "NO_TEXT"
                log(f"[+] Transcription complete for {self.vid_filepath.name}")
                
            except torch.cuda.OutOfMemoryError:
                log(f"[E] GPU out of memory for {self.vid_filepath.name}")
                self.transcript = "OOM_ERROR"
                
            except Exception as e:
                error_msg = str(e)
                # Check for known recoverable errors
                if "key.size" in error_msg or "reshape tensor of 0" in error_msg:
                    log(f"[!] Invalid audio data in {self.vid_filepath.name} (corrupt file), marking as invalid")
                    self.transcript = "CORRUPT_AUDIO"
                else:
                    log(f"[E] Exception while transcribing {self.vid_filepath.name}: {e}")
                    self.transcript = "TRANSCRIBE_ERROR"
        
            self.check_blacklisted()
            self.to_serialized()
            
            # If transcription complete and not blacklisted, queue for upload if enabled
            if ALLOW_YOUTUBE_UPLOADS and self.transcript and not self.blacklist_status and not self.youtube_uploaded:
                # Don't upload error states
                if self.transcript not in ("INVALID_FILE", "INVALID_AUDIO", "CORRUPT_AUDIO", "TRANSCRIBE_ERROR", "OOM_ERROR", "NO_TEXT"):
                    upload_queue.put(self)
                    
        except subprocess.CalledProcessError as e:
            log(f"[E] FFmpeg failed for {self.vid_filepath.name}: exit code {e.returncode}")
            self.transcript = "FFMPEG_ERROR"
            self.to_serialized()
            
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    log(f"[*] Removed temporary audio file {audio_path}")
                except Exception as e:
                    log(f"[!] Could not remove temp file {audio_path}: {e}")
            
            if lock_name in _transcribe_lock:
                _transcribe_lock.remove(lock_name)
    
    def check_blacklisted(self):
        words_in_transcript = re.findall(r'\w+', self.transcript.lower())
        blacklist_lower = [w.lower() for w in WORD_BLACKLIST]

        offending_words = [w for w in words_in_transcript if w in blacklist_lower]
        offending_words = list(dict.fromkeys(offending_words))

        self.blacklist_status = bool(offending_words)
        
        if self.blacklist_status:
            log(f"[!] Video {self.vid_filepath.name} is blacklisted due to disallowed words in transcript")
            log(f"[|] Offending: {', '.join(offending_words)}")
        
        return offending_words

    
    def to_serialized(self):
        """Writes serialized metadata to disk"""
        with fp_open(self.filepath, "wb") as f:
            self_stored = SimpleNamespace()
            self_stored.vid_filepath = self.vid_filepath
            self_stored.transcript = self.transcript
            self_stored.blacklist_status = self.blacklist_status
            self_stored.youtube_uploaded = self.youtube_uploaded
            self_stored.youtube_link = self.youtube_link
            
            self_stored.resumable_uri = self.resumable_uri
            self_stored.resumable_upload_progress = self.resumable_upload_progress
            
            pickle.dump(self_stored, f)
    
    def from_serialized(self, filepath: Path):
        try:
            with open(filepath, "rb") as f:
                self_stored = pickle.load(f)
                return self_stored
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            pass
        return None
    
    
    def upload_to_youtube(self, title=None, description=None, tags=None, category_id="20", privacy_status="private", max_retries=3):
        lock_name = self.vid_filepath.stem
        _upload_lock.append(lock_name)
        
        try:
            if not ALLOW_YOUTUBE_UPLOADS:
                log("[!] YouTube uploading is currently disabled")
                return
            
            if self.youtube_uploaded:
                log(f"[!] Already uploaded: {self.vid_filepath.name} to YouTube @ {self.youtube_link}")
                return
            
            if self.transcript == "":
                log(f"[!] Transcript missing for {self.vid_filepath.name}, will NOT upload to youtube.")
                return
            
            if self.blacklist_status:
                log(f"[!] Video {self.vid_filepath.name} is blacklisted, will NOT upload to youtube.")
                return
                
            file_size_mb = os.path.getsize(self.vid_filepath) / (1024 * 1024)
            log(f"[*] Video file size: {file_size_mb:.2f} MB")
            
            for attempt in range(max_retries):
                try:
                    log(f"[*] Uploading {self.vid_filepath.name} to youtube (attempt {attempt + 1}/{max_retries})")
                    
                    request_body = {
                        "snippet": {
                            "title": title or self.vid_filepath.stem,
                            "description": description or "Auto-Uploaded by EthanHoward/ytsync (Private Code)",
                            "tags": tags or [],
                            "categoryId": category_id
                        },
                        "status": {
                            "privacyStatus": privacy_status
                        }
                    }
                    
                    media_file = MediaFileUpload(
                        str(self.vid_filepath),
                        chunksize=1024 * 1024 * 10,  # 10MB chunks
                        resumable=True,
                        mimetype="video/mp4"
                    )

                    request = youtube.videos().insert(
                        part="snippet,status",
                        body=request_body,
                        media_body=media_file
                    )

                    if self.resumable_uri not in [None, "UPLOADED", ""]:
                        try:
                            request.resumable_uri = self.resumable_uri
                            request.resumable_progress = self.resumable_upload_progress
                            log(f"[!] Resuming upload from byte {self.resumable_upload_progress}")
                        except AttributeError as e:
                            log(f"[!] Could not resume, starting fresh: {e}")

                    response = None
                    retries = 0
                    max_chunk_retries = 5

                    while response is None:
                        try:
                            status, response = request.next_chunk()
                            if status:
                                progress = int(status.progress() * 100)
                                log(f"[*] Upload progress for {self.vid_filepath.name}: {progress}% ({status.progress() * media_file.size()}/{media_file.size()} bytes)")
                                self.resumable_uri = request.resumable_uri
                                self.resumable_upload_progress = request.resumable_progress
                                self.to_serialized()
                            retries = 0
                        except HttpError as e:
                            if e.resp.status in [500, 502, 503, 504]:
                                retries += 1
                                if retries > max_chunk_retries:
                                    log(f"[E] Too many chunk upload errors")
                                    raise

                                wait = min(2 ** retries, 64)
                                log(f"[!] Retryable error {e.resp.status}, waiting {wait}s before retry {retries}/{max_chunk_retries}")
                                sleep(wait)
                            else:
                                log(f"[E] Non-retryable error {e}")
                                raise
                        except (socket.timeout, socket.error, ConnectionError) as e:
                            retries += 1
                            if retries > max_chunk_retries:
                                log(f"[E] Too many connection errors during upload")
                                raise
                            wait = min(2 ** retries, 64)
                            log(f"[!] Connection error: {type(e).__name__}, waiting {wait}s before retry {retries}/{max_chunk_retries}")
                            sleep(wait)

                    self.youtube_uploaded = True
                    self.youtube_link = f"https://youtu.be/{response['id']}"
                    log(f"[+] Upload complete! Video ID: {response['id']} at {self.youtube_link}")
                    
                    self.resumable_uri = "UPLOADED"
                    self.resumable_upload_progress = -1

                    self.to_serialized()
                    return
                    
                except HttpError as e:
                    log(f"[E] HTTP Error {e.resp.status}: {e.content.decode('utf-8') if e.content else 'No content'}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 30
                        log(f"[!] Retrying full upload in {wait_time} seconds...")
                        sleep(wait_time)
                    else:
                        log(f"[E] Upload failed after {max_retries} attempts")
                        raise
                except (TimeoutError, ConnectionError, OSError, socket.timeout) as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 30
                        log(f"[!] Network error: {type(e).__name__}: {str(e)}")
                        log(f"[!] Retrying full upload in {wait_time} seconds...")
                        sleep(wait_time)
                    else:
                        log(f"[E] Upload failed after {max_retries} attempts")
                        raise
        finally:
            if lock_name in _upload_lock:
                _upload_lock.remove(lock_name)
        

_d_log_start = datetime.now()

def validate_video_file(video_path: Path) -> tuple[bool, str]:
    """Check if video has valid audio before processing"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0', 
             '-show_entries', 'stream=duration,codec_name', 
             '-of', 'json', str(video_path)],
            capture_output=True, 
            text=True, 
            timeout=10,
            check=False
        )
        
        if result.returncode != 0:
            return False, "No audio stream found"
        
        import json
        info = json.loads(result.stdout)
        
        if not info.get('streams'):
            return False, "No audio streams in file"
        
        duration = float(info['streams'][0].get('duration', 0))
        if duration < 0.1:
            return False, f"Audio too short: {duration:.2f}s"
        
        codec = info['streams'][0].get('codec_name', 'unknown')
        log(f"[*] Audio: {codec}, {duration:.2f}s")
        
        return True, "Valid"
        
    except subprocess.TimeoutExpired:
        return False, "FFprobe timeout"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def get_log_name() -> str:
    return f"ysdlog-{_d_log_start.strftime('%Y-%m-%d-%H.%M.%S')}.log"


def log(message: str):
    s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{s}]{message}")
    with fp_open(APPLICATION_DATA_DIRECTORY.resolve() / get_log_name(), "a+") as f:
        f.write(f"[{s}]{message}\n")
    
    
def fp_open(filepath: Path, mode: str) -> any:
    """Opens a file at a given file absolute path, makes all required directories if they do not already exist"""    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if "r" not in mode:
        open(filepath, "a").close()
    
    return open(filepath, mode)


socket.setdefaulttimeout(300)

if os.path.exists(CREDENTIALS_FILE):
    with fp_open(CREDENTIALS_FILE, "rb") as cred_file:
        credentials = pickle.load(cred_file)
        log("[*] Loaded existing credentials from file.")
else:
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    credentials = flow.run_local_server(
        port=0,
        access_type="offline",
        prompt="consent"
        )

    with fp_open(CREDENTIALS_FILE, "wb") as cred_file:
        pickle.dump(credentials, cred_file)
        log("[*] Saved new credentials to file.")

youtube = build("youtube", "v3", credentials=credentials)
    

def walk_directory(directory: Path):
    try:
        directory = directory.resolve(strict=True)
    except OSError:
        exit()
        
    items = os.listdir(directory)
    
    for item in items:
        item_path = directory / item
        if item_path.is_dir():
            yield from walk_directory(item_path)
        elif item_path.is_file() and item_path.suffix in EXTENSIONS:
            yield item_path


def extract_audio_ffmpeg(video_path: Path, output_path: Path, stereo=True) -> Path:
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-map", "0:a:0"]
    if output_path.exists():
        log(f"[!] Deleted old temporary audio file '{str(output_path)}'")
        output_path.unlink()
        
    if stereo:
        cmd += ["-ac", "2"]
        cmd += ["-c:a", "libmp3lame"]
    else:
        cmd += ["-c:a", "copy"]
    
    cmd.append(str(output_path))
    
    log(f"[*] Stripping audio with command '{cmd}'")
    
    subprocess.run(
        cmd, 
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        )
    
    return output_path


def transcribe_worker():
    """Worker thread that processes transcription queue"""
    log("[*] Transcribe worker started")
    while not exit_event.is_set():
        try:
            metadata_file = transcribe_queue.get(timeout=1)
            if metadata_file is None:
                break
            
            log(f"[*] Starting transcription for {metadata_file.vid_filepath.name}")
            metadata_file.transcribe_to_text()
            transcribe_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            log(f"[E] Error in transcribe worker: {e}")
            transcribe_queue.task_done()
    
    log("[*] Transcribe worker stopped")


def upload_worker():
    """Worker thread that processes upload queue"""
    log("[*] Upload worker started")
    while not exit_event.is_set():
        try:
            metadata_file = upload_queue.get(timeout=1)
            if metadata_file is None:
                break
            
            log(f"[*] Starting upload for {metadata_file.vid_filepath.name}")
            metadata_file.upload_to_youtube()
            upload_queue.task_done()
            print(f"[!] {len(upload_queue.queue)} Uploads Remaining in Queue")
        except queue.Empty:
            continue
        except Exception as e:
            log(f"[E] Error in upload worker: {e}")
            upload_queue.task_done()
    
    log("[*] Upload worker stopped")


def main_subroutine():
    global metadata_files
    
    files = [dir for dir in walk_directory(TARGET_DIRECTORY)]
    log(f"[*] Found {len(files)} files to process in {TARGET_DIRECTORY}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log(f"[!] Loading torch in {device} mode ({torch.cuda.is_available()})")
    
    if device != "cuda":
        log("[!] Could not load CUDA, this will be significantly slower")
        log(torch.__version__)
        log(torch.version.cuda)
        log(torch.cuda.is_available())
        log(torch.cuda.device_count())

    try:
        model = whisper.load_model(name=Path(Path.cwd() / "models" / "medium.pt").resolve()).to(device)
    except RuntimeError:
        model = whisper.load_model("medium", download_root="./models/").to(device)
    
    for f in files:
        vf_f = MetadataFile(model, f.resolve())
        metadata_files.append(vf_f)
    
    metadata_files = sorted(metadata_files, key=lambda obj: os.path.getsize(obj.vid_filepath))
    
    log(f"[~] Finished loading / importing {len(metadata_files)} files from disk")
    
    # Start worker threads
    for i in range(MAX_TRANSCRIBE_WORKERS):
        t = threading.Thread(target=transcribe_worker, daemon=True, name=f"TranscribeWorker-{i}")
        t.start()
    
    for i in range(MAX_UPLOAD_WORKERS):
        t = threading.Thread(target=upload_worker, daemon=True, name=f"UploadWorker-{i}")
        t.start()
    
    log(f"[*] Worker threads started. Transcribing: {ALLOW_TRANSCRIBING}, Uploading: {ALLOW_YOUTUBE_UPLOADS}")
    
    # Queue all work immediately based on configuration
    if ALLOW_TRANSCRIBING:
        for mf in metadata_files:
            if not mf.transcript:
                transcribe_queue.put(mf)
        log(f"[*] Queued {transcribe_queue.qsize()} videos for transcription")
    
    if ALLOW_YOUTUBE_UPLOADS:
        for mf in metadata_files:
            if not mf.youtube_uploaded and mf.transcript and not mf.blacklist_status:
                upload_queue.put(mf)
        log(f"[*] Queued {upload_queue.qsize()} videos for upload")
    
    # Wait for all work to complete
    try:
        log("[*] Processing... Press Ctrl+C to gracefully shutdown")
        transcribe_queue.join()
        upload_queue.join()
        log("[*] All processing complete!")
    except KeyboardInterrupt:
        log("[*] Interrupt received, waiting for active work to complete...")
        exit_event.set()
        
        # Wait for active uploads to finish
        while len(_upload_lock) > 0:
            log(f"[*] Waiting for {len(_upload_lock)} uploads to complete...")
            sleep(5)
    
    log("[*] Shutting down thread pools...")
    transcribe_executor.shutdown(wait=True, cancel_futures=True)
    upload_executor.shutdown(wait=True, cancel_futures=True)
    
    log("[*] Application exiting...")

if __name__ == "__main__":
    try:
        main_subroutine()
    except KeyboardInterrupt:
        print("[I] KeyboardInterrupt, Exiting...")
    except Exception as e:
        log(f"[E] Unknown Exception occurred in application flow, {e}")