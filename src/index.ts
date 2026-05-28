import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { ApiError, fal } from "@fal-ai/client";
import { mkdir, stat } from "node:fs/promises";
import { basename, dirname, extname, join, parse, resolve } from "node:path";
import { z } from "zod";

const ENDPOINT_ID = "openrouter/router/video";
const DEFAULT_FLASH_MODEL = "google/gemini-3-flash-preview";
const DEFAULT_PRO_MODEL = "google/gemini-3.1-pro-preview";
const DEFAULT_TEMPERATURE = 1;
const DEFAULT_MAX_TOKENS = 4096;
const DEFAULT_FRAME_DIR = "/private/tmp/video-perception-frames";
const MAX_BATCH_FRAMES = 24;

const DEFAULT_SYSTEM_PROMPT = [
  "You are a precise video perception assistant.",
  "Analyze the supplied video content according to the user prompt.",
  "Ground claims in visible or audible evidence, note uncertainty, and avoid inventing details.",
  "When useful, include timestamps or ordered observations.",
].join(" ");

const PRO_HINTS = [
  "forensic",
  "compare",
  "timeline",
  "timestamp",
  "timestamps",
  "frame",
  "ocr",
  "text",
  "read",
  "identify",
  "safety",
  "legal",
  "medical",
  "technical",
  "detailed",
  "подроб",
  "таймкод",
  "сравн",
  "текст",
  "прочит",
  "распоз",
  "юрид",
  "медиц",
];

type FalVideoResponse = {
  output: string;
  usage?: {
    completion_tokens?: number;
    prompt_tokens?: number;
    total_tokens?: number;
    cost?: number;
  };
};

type VideoMetadata = {
  durationSeconds: number;
  fps?: number;
  frameCount?: number;
};

type FfprobeStream = {
  index?: number;
  codec_type?: string;
  codec_name?: string;
  codec_long_name?: string;
  profile?: string;
  codec_tag_string?: string;
  width?: number;
  height?: number;
  coded_width?: number;
  coded_height?: number;
  pix_fmt?: string;
  duration?: string;
  bit_rate?: string;
  nb_frames?: string;
  avg_frame_rate?: string;
  r_frame_rate?: string;
  sample_rate?: string;
  channels?: number;
  channel_layout?: string;
  sample_fmt?: string;
  color_range?: string;
  color_space?: string;
  color_transfer?: string;
  color_primaries?: string;
  field_order?: string;
  tags?: Record<string, string>;
};

type FfprobeFormat = {
  filename?: string;
  format_name?: string;
  format_long_name?: string;
  duration?: string;
  size?: string;
  bit_rate?: string;
  nb_streams?: number;
  tags?: Record<string, string>;
};

type FfprobeFullInfo = {
  streams?: FfprobeStream[];
  format?: FfprobeFormat;
};

const server = new McpServer({
  name: "video-perception-fal",
  version: "0.1.0",
});

server.registerTool(
  "analyze_video",
  {
    title: "Analyze Video",
    description:
      "Analyze one or more videos with fal OpenRouter video models. Accepts public video URLs, YouTube links supported by Gemini, data URIs, or local video file paths.",
    inputSchema: {
      video_urls: z
        .array(z.string().min(1))
        .optional()
        .describe("Public video URLs, YouTube links, fal storage URLs, or data URIs."),
      video_paths: z
        .array(z.string().min(1))
        .optional()
        .describe("Local video file paths. The server uploads them to fal storage before analysis."),
      prompt: z
        .string()
        .min(1)
        .describe("Task-specific instruction, for example summarize, transcribe, find events, compare videos, or extract timestamps."),
      system_prompt: z
        .string()
        .optional()
        .describe("Optional system instruction. If omitted, a video perception system prompt is used."),
      model: z
        .string()
        .optional()
        .describe(
          'Model selection. Use "auto" or omit for automatic selection, "flash" for Gemini 3 Flash, "pro" for Gemini 3.1 Pro, or pass an exact OpenRouter model id.',
        ),
      temperature: z
        .number()
        .min(0)
        .max(2)
        .optional()
        .describe("Optional sampling temperature from 0 to 2. Defaults to 1."),
    },
  },
  async (input) => {
    const key = process.env.FAL_KEY ?? process.env.FAL_API_KEY;
    if (!key) {
      return toolError("Missing FAL_KEY. Set FAL_KEY in the MCP server environment.");
    }

    const videoUrls = [...(input.video_urls ?? [])];
    const videoPaths = input.video_paths ?? [];

    if (videoUrls.length === 0 && videoPaths.length === 0) {
      return toolError("Provide at least one video URL or local video path.");
    }

    fal.config({ credentials: key });

    try {
      for (const path of videoPaths) {
        videoUrls.push(await uploadLocalVideo(path));
      }

      const model = chooseModel(input.model, input.prompt);
      const request = {
        video_urls: videoUrls,
        prompt: input.prompt,
        system_prompt: input.system_prompt?.trim() || DEFAULT_SYSTEM_PROMPT,
        model,
        reasoning: true,
        temperature: input.temperature ?? DEFAULT_TEMPERATURE,
        max_tokens: DEFAULT_MAX_TOKENS,
      };

      const result = await fal.subscribe(ENDPOINT_ID, {
        input: request,
        logs: true,
        onQueueUpdate: (update) => {
          if (update.status === "IN_PROGRESS") {
            for (const log of update.logs ?? []) {
              console.error(`[fal] ${log.message}`);
            }
          }
        },
      });

      const data = result.data as FalVideoResponse;
      return {
        content: [
          {
            type: "text",
            text: formatResponse(data, {
              model,
              requestId: result.requestId,
              uploadedLocalFiles: videoPaths.length,
            }),
          },
        ],
      };
    } catch (error) {
      return toolError(formatError(error));
    }
  },
);

server.registerTool(
  "extract_frame",
  {
    title: "Extract Video Frame",
    description:
      "Extract a single still frame from a local video file and save it to disk. Useful for inspecting a precise moment without sending the full video to a model.",
    inputSchema: {
      video_path: z.string().min(1).describe("Local path to a video file."),
      mode: z
        .enum(["timestamp", "frame"])
        .optional()
        .describe('Extraction mode. Use "timestamp" for a timecode or "frame" for a zero-based frame number. Defaults to "timestamp".'),
      timestamp: z
        .string()
        .optional()
        .describe('Timestamp for mode "timestamp", for example "12.5", "00:00:12.500", or "01:02:03". Defaults to "00:00:00".'),
      frame_number: z
        .number()
        .int()
        .nonnegative()
        .optional()
        .describe('Zero-based frame number for mode "frame". For example 0 is the first frame.'),
      output_path: z
        .string()
        .optional()
        .describe("Optional output image path. Supports .png, .jpg, and .jpeg. Defaults to a generated PNG in /private/tmp/video-perception-frames."),
    },
  },
  async (input) => {
    try {
      const videoPath = resolve(input.video_path);
      await assertLocalVideoFile(videoPath);
      const metadata = await getVideoMetadata(videoPath);
      const mode = input.mode ?? "timestamp";

      const extraction = buildFrameExtraction(input, mode, metadata);
      const outputPath = resolve(input.output_path?.trim() || defaultFramePath(videoPath, extraction.label));
      if (!isSupportedImageOutput(outputPath)) {
        return toolError(`Unsupported output extension for ${basename(outputPath)}. Use png, jpg, or jpeg.`);
      }

      await mkdir(dirname(outputPath), { recursive: true });
      await runFfmpeg(extraction.ffmpegArgs(videoPath, outputPath));

      return {
        content: [
          {
            type: "text",
            text: [
              `frame_path: ${outputPath}`,
              `video_path: ${videoPath}`,
              `mode: ${mode}`,
              extraction.report,
              `duration_seconds: ${formatSeconds(metadata.durationSeconds)}`,
              metadata.fps ? `fps: ${formatSeconds(metadata.fps)}` : undefined,
              typeof metadata.frameCount === "number" ? `frame_count: ${metadata.frameCount}` : undefined,
            ]
              .filter(Boolean)
              .join("\n"),
          },
        ],
      };
    } catch (error) {
      return toolError(formatError(error));
    }
  },
);

server.registerTool(
  "extract_frames",
  {
    title: "Extract Video Frames",
    description:
      "Extract multiple still frames from a local video file and save them to disk. Supports timestamp lists or zero-based frame number lists.",
    inputSchema: {
      video_path: z.string().min(1).describe("Local path to a video file."),
      mode: z
        .enum(["timestamp", "frame"])
        .optional()
        .describe('Extraction mode. Use "timestamp" with timestamps or "frame" with frame_numbers. Defaults to "timestamp".'),
      timestamps: z
        .array(z.string().min(1))
        .optional()
        .describe('Timestamps for mode "timestamp", for example ["00:00:01", "00:00:05.500", "12.5"].'),
      frame_numbers: z
        .array(z.number().int().nonnegative())
        .optional()
        .describe('Zero-based frame numbers for mode "frame", for example [0, 120, 240].'),
      output_dir: z
        .string()
        .optional()
        .describe("Optional output directory. Defaults to /private/tmp/video-perception-frames."),
      image_format: z
        .enum(["png", "jpg", "jpeg"])
        .optional()
        .describe('Output image format. Defaults to "png".'),
    },
  },
  async (input) => {
    try {
      const videoPath = resolve(input.video_path);
      await assertLocalVideoFile(videoPath);
      const metadata = await getVideoMetadata(videoPath);
      const mode = input.mode ?? "timestamp";
      const outputDir = resolve(input.output_dir?.trim() || DEFAULT_FRAME_DIR);
      const imageFormat = input.image_format ?? "png";
      await mkdir(outputDir, { recursive: true });

      const items = buildBatchItems(input, mode);
      if (items.length > MAX_BATCH_FRAMES) {
        return toolError(`Too many frames requested: ${items.length}. Maximum is ${MAX_BATCH_FRAMES} per call.`);
      }

      const results = [];
      for (const item of items) {
        try {
          const extraction = buildFrameExtraction(item.input, mode, metadata);
          const outputPath = join(outputDir, defaultFrameFileName(videoPath, extraction.label, imageFormat));
          await runFfmpeg(extraction.ffmpegArgs(videoPath, outputPath));
          results.push({ index: item.index, ok: true, path: outputPath, detail: extraction.report });
        } catch (error) {
          results.push({ index: item.index, ok: false, error: formatError(error) });
        }
      }

      const successCount = results.filter((result) => result.ok).length;
      return {
        isError: successCount === 0,
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                video_path: videoPath,
                mode,
                output_dir: outputDir,
                image_format: imageFormat,
                duration_seconds: metadata.durationSeconds,
                fps: metadata.fps,
                frame_count: metadata.frameCount,
                success_count: successCount,
                failure_count: results.length - successCount,
                results,
              },
              null,
              2,
            ),
          },
        ],
      };
    } catch (error) {
      return toolError(formatError(error));
    }
  },
);

server.registerTool(
  "video_info",
  {
    title: "Video Info",
    description:
      "Read technical metadata for a local video file using ffprobe: container, duration, size, bitrate, streams, codecs, fps, resolution, audio tracks, color space, and metadata tags.",
    inputSchema: {
      video_path: z.string().min(1).describe("Local path to a video file."),
    },
  },
  async (input) => {
    try {
      const videoPath = resolve(input.video_path);
      await assertLocalVideoFile(videoPath);
      const info = await getVideoInfo(videoPath);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(info, null, 2),
          },
        ],
      };
    } catch (error) {
      return toolError(formatError(error));
    }
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);

async function uploadLocalVideo(path: string): Promise<string> {
  const absolutePath = resolve(path);
  await assertLocalVideoFile(absolutePath);

  const file = Bun.file(absolutePath);
  return fal.storage.upload(file, {
    lifecycle: {
      expiresIn: "7d",
    },
  });
}

async function assertLocalVideoFile(path: string): Promise<void> {
  const fileStat = await stat(path);
  if (!fileStat.isFile()) {
    throw new Error(`Not a file: ${path}`);
  }

  if (!isSupportedVideoFile(path)) {
    throw new Error(`Unsupported video extension for ${basename(path)}. Use mp4, mpeg, mov, or webm.`);
  }
}

function isSupportedVideoFile(path: string): boolean {
  return [".mp4", ".mpeg", ".mpg", ".mov", ".webm"].includes(extname(path).toLowerCase());
}

function isSupportedImageOutput(path: string): boolean {
  return [".png", ".jpg", ".jpeg"].includes(extname(path).toLowerCase());
}

function defaultFramePath(videoPath: string, timestamp: string): string {
  return join(DEFAULT_FRAME_DIR, defaultFrameFileName(videoPath, timestamp, "png"));
}

function defaultFrameFileName(videoPath: string, label: string, imageFormat: "png" | "jpg" | "jpeg"): string {
  const parsed = parse(videoPath);
  const safeLabel = label.replace(/[^a-zA-Z0-9.-]+/g, "_").replace(/^_+|_+$/g, "") || "start";
  return `${parsed.name}-${safeLabel}.${imageFormat}`;
}

function buildBatchItems(
  input: { timestamps?: string[]; frame_numbers?: number[] },
  mode: "timestamp" | "frame",
): Array<{ index: number; input: { timestamp?: string; frame_number?: number } }> {
  if (mode === "frame") {
    if (!input.frame_numbers?.length) {
      throw new Error('Missing frame_numbers for mode "frame". Provide a non-empty array, for example frame_numbers: [0, 120, 240].');
    }

    return input.frame_numbers.map((frameNumber, index) => ({
      index,
      input: { frame_number: frameNumber },
    }));
  }

  if (!input.timestamps?.length) {
    throw new Error('Missing timestamps for mode "timestamp". Provide a non-empty array, for example timestamps: ["00:00:01", "00:00:05"].');
  }

  return input.timestamps.map((timestamp, index) => ({
    index,
    input: { timestamp },
  }));
}

function buildFrameExtraction(
  input: { timestamp?: string; frame_number?: number },
  mode: "timestamp" | "frame",
  metadata: VideoMetadata,
): {
  label: string;
  report: string;
  ffmpegArgs: (videoPath: string, outputPath: string) => string[];
} {
  if (mode === "frame") {
    if (typeof input.frame_number !== "number") {
      throw new Error('Missing frame_number for mode "frame". Provide a zero-based integer frame number, for example frame_number: 120.');
    }

    if (typeof metadata.frameCount === "number" && input.frame_number >= metadata.frameCount) {
      throw new Error(
        `Frame ${input.frame_number} does not exist. Valid frame range is 0..${metadata.frameCount - 1}. ` +
          `Video duration is ${formatSeconds(metadata.durationSeconds)}s${metadata.fps ? ` at about ${formatSeconds(metadata.fps)} fps` : ""}.`,
      );
    }

    return {
      label: `frame-${input.frame_number}`,
      report: `frame_number: ${input.frame_number}`,
      ffmpegArgs: (videoPath, outputPath) => [
        "-y",
        "-i",
        videoPath,
        "-vf",
        `select=eq(n\\,${input.frame_number})`,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        outputPath,
      ],
    };
  }

  const timestamp = input.timestamp?.trim() || "00:00:00";
  const seconds = parseTimestampSeconds(timestamp);
  if (seconds >= metadata.durationSeconds) {
    throw new Error(
      `Timestamp ${timestamp} (${formatSeconds(seconds)}s) is outside this video. ` +
        `Valid timestamp range is 00:00:00 through ${secondsToTimestamp(Math.max(0, metadata.durationSeconds - 0.001))} ` +
        `(${formatSeconds(metadata.durationSeconds)}s duration).`,
    );
  }

  return {
    label: timestamp,
    report: `timestamp: ${timestamp}`,
    ffmpegArgs: (videoPath, outputPath) => [
      "-y",
      "-ss",
      timestamp,
      "-i",
      videoPath,
      "-frames:v",
      "1",
      "-q:v",
      "2",
      outputPath,
    ],
  };
}

function parseTimestampSeconds(timestamp: string): number {
  if (/^\d+(\.\d+)?$/.test(timestamp)) {
    return Number(timestamp);
  }

  const match = timestamp.match(/^(\d{1,2}):([0-5]\d):([0-5]\d(?:\.\d+)?)$/);
  if (!match) {
    throw new Error(
      `Invalid timestamp "${timestamp}". Use seconds like "12.5" or timecode "HH:MM:SS.mmm", for example "00:00:12.500".`,
    );
  }

  const [, hours, minutes, seconds] = match;
  return Number(hours) * 3600 + Number(minutes) * 60 + Number(seconds);
}

function secondsToTimestamp(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${seconds.toFixed(3).padStart(6, "0")}`;
}

function formatSeconds(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

async function getVideoMetadata(videoPath: string): Promise<VideoMetadata> {
  const output = await runFfprobe([
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=duration,nb_frames,avg_frame_rate,r_frame_rate",
    "-show_entries",
    "format=duration",
    "-of",
    "json",
    videoPath,
  ]);
  const parsed = JSON.parse(output) as {
    streams?: Array<{ duration?: string; nb_frames?: string; avg_frame_rate?: string; r_frame_rate?: string }>;
    format?: { duration?: string };
  };
  const stream = parsed.streams?.[0];
  const durationSeconds = Number(stream?.duration ?? parsed.format?.duration);
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    throw new Error("Could not read video duration with ffprobe.");
  }

  const fps = parseFrameRate(stream?.avg_frame_rate) ?? parseFrameRate(stream?.r_frame_rate);
  const explicitFrameCount = stream?.nb_frames && stream.nb_frames !== "N/A" ? Number(stream.nb_frames) : undefined;
  const frameCount =
    typeof explicitFrameCount === "number" && Number.isFinite(explicitFrameCount)
      ? explicitFrameCount
      : fps
        ? Math.max(1, Math.floor(durationSeconds * fps))
        : undefined;

  return { durationSeconds, fps, frameCount };
}

async function getVideoInfo(videoPath: string) {
  const output = await runFfprobe([
    "-v",
    "error",
    "-show_format",
    "-show_streams",
    "-of",
    "json",
    videoPath,
  ]);
  const parsed = JSON.parse(output) as FfprobeFullInfo;
  const format = parsed.format;
  const streams = parsed.streams ?? [];

  return {
    path: videoPath,
    container: format
      ? {
          format_name: format.format_name,
          format_long_name: format.format_long_name,
          duration_seconds: numberFromString(format.duration),
          size_bytes: integerFromString(format.size),
          bit_rate: integerFromString(format.bit_rate),
          stream_count: format.nb_streams,
          tags: format.tags ?? {},
        }
      : undefined,
    streams: streams.map((stream) => normalizeStreamInfo(stream)),
    video_streams: streams.filter((stream) => stream.codec_type === "video").map((stream) => normalizeStreamInfo(stream)),
    audio_streams: streams.filter((stream) => stream.codec_type === "audio").map((stream) => normalizeStreamInfo(stream)),
    other_streams: streams
      .filter((stream) => stream.codec_type !== "video" && stream.codec_type !== "audio")
      .map((stream) => normalizeStreamInfo(stream)),
  };
}

function normalizeStreamInfo(stream: FfprobeStream) {
  const base = {
    index: stream.index,
    type: stream.codec_type,
    codec_name: stream.codec_name,
    codec_long_name: stream.codec_long_name,
    profile: stream.profile,
    codec_tag: stream.codec_tag_string,
    duration_seconds: numberFromString(stream.duration),
    bit_rate: integerFromString(stream.bit_rate),
    tags: stream.tags ?? {},
  };

  if (stream.codec_type === "video") {
    return {
      ...base,
      width: stream.width,
      height: stream.height,
      coded_width: stream.coded_width,
      coded_height: stream.coded_height,
      pixel_format: stream.pix_fmt,
      fps: parseFrameRate(stream.avg_frame_rate) ?? parseFrameRate(stream.r_frame_rate),
      avg_frame_rate: stream.avg_frame_rate,
      real_frame_rate: stream.r_frame_rate,
      frame_count: integerFromString(stream.nb_frames),
      color_range: stream.color_range,
      color_space: stream.color_space,
      color_transfer: stream.color_transfer,
      color_primaries: stream.color_primaries,
      field_order: stream.field_order,
    };
  }

  if (stream.codec_type === "audio") {
    return {
      ...base,
      sample_rate: integerFromString(stream.sample_rate),
      channels: stream.channels,
      channel_layout: stream.channel_layout,
      sample_format: stream.sample_fmt,
      frame_count: integerFromString(stream.nb_frames),
    };
  }

  return base;
}

function numberFromString(value: string | undefined): number | undefined {
  if (!value || value === "N/A") {
    return undefined;
  }

  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function integerFromString(value: string | undefined): number | undefined {
  const number = numberFromString(value);
  return typeof number === "number" ? Math.trunc(number) : undefined;
}

function parseFrameRate(value: string | undefined): number | undefined {
  if (!value || value === "0/0") {
    return undefined;
  }

  const [numerator, denominator] = value.split("/").map(Number);
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) {
    return undefined;
  }

  return numerator / denominator;
}

async function runFfprobe(args: string[]): Promise<string> {
  const proc = Bun.spawn(["ffprobe", ...args], {
    stdout: "pipe",
    stderr: "pipe",
  });

  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);

  if (exitCode !== 0) {
    throw new Error(`ffprobe failed with exit code ${exitCode}\n${stderr || stdout}`);
  }

  return stdout;
}

async function runFfmpeg(args: string[]): Promise<void> {
  const proc = Bun.spawn(["ffmpeg", ...args], {
    stdout: "pipe",
    stderr: "pipe",
  });

  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);

  if (exitCode !== 0) {
    throw new Error(`ffmpeg failed with exit code ${exitCode}\n${stderr || stdout}`);
  }
}

function chooseModel(model: string | undefined, prompt: string): string {
  const normalized = model?.trim();
  if (!normalized || normalized === "auto") {
    const lowerPrompt = prompt.toLowerCase();
    return PRO_HINTS.some((hint) => lowerPrompt.includes(hint)) ? DEFAULT_PRO_MODEL : DEFAULT_FLASH_MODEL;
  }

  if (normalized === "flash") {
    return DEFAULT_FLASH_MODEL;
  }

  if (normalized === "pro") {
    return DEFAULT_PRO_MODEL;
  }

  return normalized;
}

function formatResponse(
  data: FalVideoResponse,
  meta: { model: string; requestId?: string; uploadedLocalFiles: number },
): string {
  const usage = data.usage
    ? [
        typeof data.usage.prompt_tokens === "number" ? `prompt=${data.usage.prompt_tokens}` : undefined,
        typeof data.usage.completion_tokens === "number" ? `completion=${data.usage.completion_tokens}` : undefined,
        typeof data.usage.total_tokens === "number" ? `total=${data.usage.total_tokens}` : undefined,
        typeof data.usage.cost === "number" ? `cost=${data.usage.cost}` : undefined,
      ]
        .filter(Boolean)
        .join(", ")
    : "not reported";

  return [
    data.output,
    "",
    "---",
    `model: ${meta.model}`,
    meta.requestId ? `request_id: ${meta.requestId}` : undefined,
    `uploaded_local_files: ${meta.uploadedLocalFiles}`,
    `usage: ${usage}`,
  ]
    .filter(Boolean)
    .join("\n");
}

function toolError(message: string) {
  return {
    isError: true,
    content: [
      {
        type: "text" as const,
        text: message,
      },
    ],
  };
}

function formatError(error: unknown): string {
  if (error instanceof ApiError) {
    const detail = error.body ? `\n${JSON.stringify(error.body, null, 2)}` : "";
    const requestId = error.requestId ? `\nrequest_id: ${error.requestId}` : "";
    return `${error.message} (${error.status})${requestId}${detail}`;
  }

  return error instanceof Error ? error.message : String(error);
}
