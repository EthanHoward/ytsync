# ytsync credential regeneration and matching guide

## Regenerating expired credentials

1. Stop any running `main.py` process.
2. Ensure `conf.ini` points to:
   - `YouTube.client_secrets_file` (OAuth desktop client JSON from Google Cloud Console)
   - `Paths.credentials_file` (token pickle file path)
3. Run `python main.py`.
4. If the refresh token is still valid, ytsync now refreshes automatically.
5. If refresh fails or credentials are invalid, ytsync automatically starts a browser OAuth login and writes a new credential file.

### Forced clean re-auth (manual reset)

If you want to force a full re-consent:

1. Delete the credentials file configured in `Paths.credentials_file`.
2. Run `python main.py`.
3. Complete browser consent.

## Better file catalog strategy (rename-safe)

ytsync now embeds a metadata block in the uploaded YouTube description:

- `local_id`: a stable hash fingerprint based on file size + first bytes of the file
- `relative_path`: original path relative to `target_directory`

This means matching is still possible even if the YouTube title changes, because title is no longer the identity key.

## "Not made for kids" automation

Uploads now set `status.selfDeclaredMadeForKids=false` on each `videos.insert` request.

## Extra speed recommendations

1. Keep resumable uploads enabled (already done).
2. Use GPU + CUDA for whisper.
3. If transcription quality allows, switch from `medium` whisper model to `small` for faster throughput.
4. Store whisper model file locally (`./models`) to avoid startup downloads.
5. Increase `MAX_TRANSCRIBE_WORKERS` only if your CPU/GPU and I/O can handle it without contention.
