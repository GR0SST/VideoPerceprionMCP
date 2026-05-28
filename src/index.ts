import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { ApiError, fal } from "@fal-ai/client";
import { stat } from "node:fs/promises";
import { basename, extname, resolve } from "node:path";
import { z } from "zod";

const ENDPOINT_ID = "openrouter/router/video";
const DEFAULT_FLASH_MODEL = "google/gemini-3-flash-preview";
const DEFAULT_PRO_MODEL = "google/gemini-3.1-pro-preview";
const DEFAULT_TEMPERATURE = 1;
const DEFAULT_MAX_TOKENS = 4096;

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

const transport = new StdioServerTransport();
await server.connect(transport);

async function uploadLocalVideo(path: string): Promise<string> {
  const absolutePath = resolve(path);
  const fileStat = await stat(absolutePath);
  if (!fileStat.isFile()) {
    throw new Error(`Not a file: ${absolutePath}`);
  }

  const file = Bun.file(absolutePath);
  if (!isSupportedVideoFile(absolutePath)) {
    throw new Error(`Unsupported video extension for ${basename(absolutePath)}. Use mp4, mpeg, mov, or webm.`);
  }

  return fal.storage.upload(file, {
    lifecycle: {
      expiresIn: "7d",
    },
  });
}

function isSupportedVideoFile(path: string): boolean {
  return [".mp4", ".mpeg", ".mpg", ".mov", ".webm"].includes(extname(path).toLowerCase());
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
