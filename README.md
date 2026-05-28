# Video Perception MCP

Bun-based MCP server for analyzing video with fal's OpenRouter video endpoint.

## Features

- MCP tools: `analyze_video`, `extract_frame`
- Accepts public video URLs, supported YouTube links, data URIs, or local video paths
- Uploads local files to fal storage automatically
- Uses `google/gemini-3-flash-preview` by default
- Can switch to `google/gemini-3.1-pro-preview` with `model: "pro"` or automatic prompt heuristics
- Supports optional `system_prompt` and `temperature`
- Always sends `reasoning: true` because the fal video endpoint currently requires it
- Uses an internal `max_tokens` limit of `4096`
- Extracts local video frames to disk with `ffmpeg`

## Setup

```bash
bun install
export FAL_KEY="your_fal_api_key"
bun run start
```

## MCP Client Config

Use an absolute path for the project directory:

```json
{
  "mcpServers": {
    "video-perception": {
      "command": "bun",
      "args": ["src/index.ts"],
      "cwd": "/path/to/this/repository/VideoPerceprionMCP",
      "env": {
        "FAL_KEY": "your_fal_api_key"
      }
    }
  }
}
```

## Tool Input Examples

Analyze a URL:

```json
{
  "video_urls": ["https://example.com/video.mp4"],
  "prompt": "Describe the scene and list important events with timestamps."
}
```

Analyze a local file:

```json
{
  "video_paths": ["/absolute/path/to/video.mp4"],
  "prompt": "Transcribe speech and summarize visible actions.",
  "model": "pro"
}
```

Extract one frame:

```json
{
  "video_path": "/absolute/path/to/video.mp4",
  "mode": "timestamp",
  "timestamp": "00:00:05.000",
  "output_path": "/private/tmp/frame.png"
}
```

Extract by zero-based frame number:

```json
{
  "video_path": "/absolute/path/to/video.mp4",
  "mode": "frame",
  "frame_number": 120
}
```
