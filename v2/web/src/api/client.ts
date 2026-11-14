import type { z } from 'zod';

import {
  type ChatRequest,
  type ChatResponse,
  ChatResponseSchema,
  type ConfigResponse,
  ConfigResponseSchema,
  ErrorResponseSchema,
  type HealthResponse,
  HealthResponseSchema,
  type InputLang,
  type Redaction,
  StreamDoneSchema,
  StreamErrorSchema,
  type StreamMeta,
  StreamMetaSchema,
  StreamTokenSchema,
  type TranscribeResponse,
  TranscribeResponseSchema,
} from './schemas';
import { createSseParser } from './sse';

/** Every failure the client can produce, normalised to one typed error. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | undefined;
  readonly retryAfterSeconds: number | undefined;

  constructor(opts: {
    status: number;
    code: string;
    message: string;
    requestId?: string | undefined;
    retryAfter?: number | undefined;
  }) {
    super(opts.message);
    this.name = 'ApiError';
    this.status = opts.status;
    this.code = opts.code;
    this.requestId = opts.requestId;
    this.retryAfterSeconds = opts.retryAfter;
  }

  /** Worth retrying automatically or via a Retry button. */
  get retryable(): boolean {
    return this.status === 0 || this.status === 429 || this.status >= 500;
  }
}

export type StreamHandlers = {
  onMeta: (meta: StreamMeta) => void;
  onToken: (delta: string) => void;
  onDone: (redactions: Redaction[]) => void;
};

export type ApiClient = ReturnType<typeof createApiClient>;

type Options = { fetch?: typeof fetch; timeoutMs?: number };

const DEFAULT_TIMEOUT_MS = 60_000;

function withTimeout(
  signal: AbortSignal | undefined,
  ms: number,
): { signal: AbortSignal; clear: () => void } {
  const ctrl = new AbortController();
  const timer = setTimeout(() => {
    ctrl.abort(new DOMException('timeout', 'TimeoutError'));
  }, ms);
  const onAbort = () => {
    ctrl.abort(signal?.reason);
  };
  if (signal?.aborted) ctrl.abort(signal.reason);
  signal?.addEventListener('abort', onAbort, { once: true });
  return {
    signal: ctrl.signal,
    clear: () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
    },
  };
}

function isAbort(e: unknown): e is DOMException {
  return e instanceof DOMException && (e.name === 'AbortError' || e.name === 'TimeoutError');
}

async function toApiError(res: Response): Promise<ApiError> {
  const retry = Number(res.headers.get('retry-after'));
  const base = {
    status: res.status,
    requestId: res.headers.get('x-request-id') ?? undefined,
    ...(Number.isFinite(retry) && retry > 0 ? { retryAfter: retry } : {}),
  };
  try {
    const parsed = ErrorResponseSchema.safeParse(await res.json());
    if (parsed.success) {
      return new ApiError({ ...base, ...parsed.data.error, requestId: parsed.data.error.request_id });
    }
  } catch {
    /* non-JSON error body */
  }
  return new ApiError({ ...base, code: `http_${res.status}`, message: `Request failed (${res.status}).` });
}

function parseOrThrow<S extends z.ZodType>(schema: S, value: unknown, what: string): z.infer<S> {
  const r = schema.safeParse(value);
  if (!r.success) {
    throw new ApiError({
      status: 0,
      code: 'contract_mismatch',
      message: `Unexpected ${what} from server.`,
    });
  }
  return r.data;
}

export function createApiClient(baseUrl: string, opts: Options = {}) {
  const doFetch = opts.fetch ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  async function send(
    path: string,
    init: RequestInit,
    signal?: AbortSignal,
    ms = timeoutMs,
  ): Promise<Response> {
    const t = withTimeout(signal, ms);
    try {
      const res = await doFetch(`${baseUrl}${path}`, { ...init, signal: t.signal, credentials: 'omit' });
      if (!res.ok) throw await toApiError(res);
      return res;
    } catch (e) {
      if (e instanceof ApiError) throw e;
      if (isAbort(e) || t.signal.aborted) {
        const reason: unknown = t.signal.reason;
        if (reason instanceof DOMException && reason.name === 'TimeoutError') {
          throw new ApiError({ status: 0, code: 'timeout', message: 'The server took too long to respond.' });
        }
        throw e; // caller-initiated abort: propagate as-is
      }
      throw new ApiError({ status: 0, code: 'network_error', message: 'Could not reach the server.' });
    } finally {
      t.clear();
    }
  }

  async function getJson<S extends z.ZodType>(
    path: string,
    schema: S,
    signal?: AbortSignal,
  ): Promise<z.infer<S>> {
    const res = await send(path, { method: 'GET', headers: { Accept: 'application/json' } }, signal);
    return parseOrThrow(schema, await res.json(), path);
  }

  return {
    getHealth: (signal?: AbortSignal): Promise<HealthResponse> =>
      getJson('/health', HealthResponseSchema, signal),

    getConfig: (signal?: AbortSignal): Promise<ConfigResponse> =>
      getJson('/api/config', ConfigResponseSchema, signal),

    async chat(req: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
      const res = await send(
        '/api/chat',
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req) },
        signal,
      );
      return parseOrThrow(ChatResponseSchema, await res.json(), 'chat response');
    },

    /**
     * Streams an answer. Resolves when the `done` event arrives; rejects with
     * ApiError on HTTP errors, `error` events, or a stream that ends early.
     * `timeoutMs` applies to the time-to-headers and to each gap between chunks.
     */
    async chatStream(req: ChatRequest, handlers: StreamHandlers, signal?: AbortSignal): Promise<void> {
      const res = await send(
        '/api/chat/stream',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
          body: JSON.stringify(req),
        },
        signal,
        timeoutMs,
      );
      if (!res.body)
        throw new ApiError({ status: 0, code: 'stream_unsupported', message: 'Streaming unsupported.' });

      // Mutated from the parser callback; an object so control-flow narrowing doesn't assume constants.
      const st: { done: boolean; failure: ApiError | undefined } = { done: false, failure: undefined };
      const parser = createSseParser(({ event, data }) => {
        if (st.done || st.failure) return;
        let payload: unknown;
        try {
          payload = JSON.parse(data);
        } catch {
          st.failure = new ApiError({
            status: 0,
            code: 'contract_mismatch',
            message: 'Malformed stream event.',
          });
          return;
        }
        if (event === 'meta') handlers.onMeta(parseOrThrow(StreamMetaSchema, payload, 'stream metadata'));
        else if (event === 'token')
          handlers.onToken(parseOrThrow(StreamTokenSchema, payload, 'stream token').delta);
        else if (event === 'done') {
          st.done = true;
          handlers.onDone(parseOrThrow(StreamDoneSchema, payload, 'stream end').redactions);
        } else if (event === 'error') {
          const e = parseOrThrow(StreamErrorSchema, payload, 'stream error');
          st.failure = new ApiError({
            status: 502,
            code: e.code,
            message: e.message,
            requestId: e.request_id,
          });
        }
      });

      // `send` only guards until headers arrive. From here the stream needs its
      // own lifecycle: honour caller aborts and fail if the server goes silent.
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
      const abortReason = () => signal?.reason as unknown;
      const onAbort = () => {
        void reader.cancel(abortReason()).catch(() => undefined);
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      let idle: ReturnType<typeof setTimeout> | undefined;
      const readWithIdleTimeout = () =>
        new Promise<ReadableStreamReadResult<string>>((resolve, reject) => {
          idle = setTimeout(() => {
            reject(new ApiError({ status: 0, code: 'timeout', message: 'The answer stalled.' }));
            void reader.cancel().catch(() => undefined);
          }, timeoutMs);
          reader.read().then(resolve, reject);
        }).finally(() => {
          clearTimeout(idle);
        });
      try {
        for (;;) {
          if (signal?.aborted)
            throw signal.reason instanceof DOMException
              ? signal.reason
              : new DOMException('Aborted', 'AbortError');
          const { value, done: eof } = await readWithIdleTimeout();
          if (eof) {
            if (signal?.aborted) continue; // re-check at loop top → AbortError
            break;
          }
          parser.feed(value);
          if (st.failure) throw st.failure;
          if (st.done) break;
        }
        parser.end();
        if (st.failure) throw st.failure;
      } catch (e) {
        if (e instanceof ApiError || isAbort(e)) throw e;
        throw new ApiError({ status: 0, code: 'network_error', message: 'The connection was interrupted.' });
      } finally {
        signal?.removeEventListener('abort', onAbort);
        reader.releaseLock();
      }
      if (!st.done) {
        throw new ApiError({ status: 0, code: 'stream_incomplete', message: 'The answer was cut off.' });
      }
    },

    async transcribe(audio: Blob, inputLang: InputLang, signal?: AbortSignal): Promise<TranscribeResponse> {
      const form = new FormData();
      const ext = audio.type.includes('ogg') ? 'ogg' : audio.type.includes('mp4') ? 'm4a' : 'webm';
      form.append('audio', new File([audio], `recording.${ext}`, { type: audio.type }));
      form.append('input_lang', inputLang);
      const res = await send('/api/transcribe', { method: 'POST', body: form }, signal, timeoutMs * 2);
      return parseOrThrow(TranscribeResponseSchema, await res.json(), 'transcription');
    },

    async tts(text: string, lang: 'en' | 'hi', signal?: AbortSignal): Promise<Blob> {
      const res = await send(
        '/api/tts',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, lang }),
        },
        signal,
      );
      return res.blob();
    },
  };
}
