import { http, HttpResponse } from 'msw';

import {
  API,
  chatFixture,
  chunkedStream,
  configFixture,
  errorBody,
  metaFixture,
  sse,
  streamOk,
} from '../test/fixtures';
import { server } from '../test/server';
import { ApiError, createApiClient } from './client';
import type { ChatRequest } from './schemas';

const api = createApiClient(API);
const req: ChatRequest = {
  message: 'What is the President paid?',
  input_lang: 'auto',
  output_lang: 'en',
  history: [],
};

async function rejection(p: Promise<unknown>): Promise<ApiError> {
  try {
    await p;
  } catch (e) {
    if (e instanceof ApiError) return e;
    throw e;
  }
  throw new Error('expected rejection');
}

describe('api client: JSON endpoints', () => {
  it('parses config and health', async () => {
    await expect(api.getConfig()).resolves.toEqual(configFixture);
    await expect(api.getHealth()).resolves.toMatchObject({ status: 'ok', index_ready: true });
  });

  it('sends the chat request body and parses the response', async () => {
    let body: unknown;
    server.use(
      http.post(`${API}/api/chat`, async ({ request }) => {
        body = await request.json();
        return HttpResponse.json(chatFixture);
      }),
    );
    await expect(api.chat(req)).resolves.toEqual(chatFixture);
    expect(body).toEqual(req);
  });

  it('never sends cookies or credentials', async () => {
    const seen: RequestInit[] = [];
    const spy = createApiClient(API, {
      fetch: (input, init) => {
        seen.push(init ?? {});
        return fetch(input, init);
      },
    });
    await spy.getHealth();
    expect(seen[0]?.credentials).toBe('omit');
  });

  it('maps backend error bodies to typed ApiError with retry-after', async () => {
    server.use(
      http.post(`${API}/api/chat`, () =>
        HttpResponse.json(errorBody('rate_limited', 'Too many requests.'), {
          status: 429,
          headers: { 'Retry-After': '12' },
        }),
      ),
    );
    const e = await rejection(api.chat(req));
    expect(e).toMatchObject({
      status: 429,
      code: 'rate_limited',
      retryAfterSeconds: 12,
      requestId: 'req-err00001',
    });
    expect(e.retryable).toBe(true);
  });

  it('treats validation errors as non-retryable', async () => {
    server.use(
      http.post(`${API}/api/chat`, () =>
        HttpResponse.json(errorBody('message_too_long', 'Too long.'), { status: 422 }),
      ),
    );
    const e = await rejection(api.chat(req));
    expect(e.code).toBe('message_too_long');
    expect(e.retryable).toBe(false);
  });

  it('handles non-JSON error bodies', async () => {
    server.use(
      http.get(`${API}/health`, () => new HttpResponse('<html>Bad Gateway</html>', { status: 502 })),
    );
    const e = await rejection(api.getHealth());
    expect(e).toMatchObject({ status: 502, code: 'http_502' });
  });

  it('reports network failures', async () => {
    server.use(http.get(`${API}/health`, () => HttpResponse.error()));
    expect((await rejection(api.getHealth())).code).toBe('network_error');
  });

  it('rejects responses that do not match the contract', async () => {
    server.use(
      http.get(`${API}/api/config`, () => HttpResponse.json({ ...configFixture, tts_enabled: 'yes' })),
    );
    expect((await rejection(api.getConfig())).code).toBe('contract_mismatch');
  });

  it('times out slow requests', async () => {
    server.use(http.get(`${API}/health`, () => new Promise<never>(() => {})));
    const e = await rejection(createApiClient(API, { timeoutMs: 30 }).getHealth());
    expect(e.code).toBe('timeout');
  });

  it('propagates caller aborts unchanged', async () => {
    server.use(http.get(`${API}/health`, () => new Promise<never>(() => {})));
    const ctrl = new AbortController();
    const p = api.getHealth(ctrl.signal);
    ctrl.abort();
    await expect(p).rejects.toMatchObject({ name: 'AbortError' });
  });
});

describe('api client: streaming', () => {
  const collect = () => {
    const tokens: string[] = [];
    const calls: string[] = [];
    return {
      tokens,
      calls,
      handlers: {
        onMeta: () => calls.push('meta'),
        onToken: (d: string) => {
          tokens.push(d);
          calls.push('token');
        },
        onDone: () => calls.push('done'),
      },
    };
  };

  it('delivers meta, tokens and done in order across chunk boundaries', async () => {
    const c = collect();
    await api.chatStream(req, c.handlers);
    expect(c.tokens.join('')).toBe('The President receives $400,000 a year [S1].');
    expect(c.calls[0]).toBe('meta');
    expect(c.calls.at(-1)).toBe('done');
  });

  it('turns an error event into an ApiError', async () => {
    server.use(
      http.post(
        `${API}/api/chat/stream`,
        () =>
          new HttpResponse(
            sse([
              ['meta', metaFixture],
              ['token', { delta: 'The Pres' }],
              ['error', { code: 'stream_failed', message: 'Could not finish.', request_id: 'r-12345678' }],
            ]),
          ),
      ),
    );
    const c = collect();
    const e = await rejection(api.chatStream(req, c.handlers));
    expect(e).toMatchObject({ code: 'stream_failed', requestId: 'r-12345678' });
    expect(e.retryable).toBe(true);
    expect(c.tokens).toEqual(['The Pres']);
  });

  it('detects a stream that ends without done', async () => {
    server.use(
      http.post(
        `${API}/api/chat/stream`,
        () => new HttpResponse(chunkedStream(sse([['meta', metaFixture]]))),
      ),
    );
    expect((await rejection(api.chatStream(req, collect().handlers))).code).toBe('stream_incomplete');
  });

  it('rejects malformed events', async () => {
    server.use(
      http.post(`${API}/api/chat/stream`, () => new HttpResponse('event: meta\ndata: {not json\n\n')),
    );
    expect((await rejection(api.chatStream(req, collect().handlers))).code).toBe('contract_mismatch');
  });

  it('surfaces HTTP errors before the stream starts', async () => {
    server.use(
      http.post(`${API}/api/chat/stream`, () =>
        HttpResponse.json(errorBody('budget_exhausted', 'Daily limit reached.'), { status: 503 }),
      ),
    );
    expect((await rejection(api.chatStream(req, collect().handlers))).code).toBe('budget_exhausted');
  });

  it('can be cancelled mid-stream', async () => {
    server.use(
      http.post(`${API}/api/chat/stream`, () => {
        const enc = new TextEncoder();
        return new HttpResponse(
          new ReadableStream({
            start(ctrl) {
              ctrl.enqueue(enc.encode(sse([['meta', metaFixture]])));
            },
          }),
        );
      }),
    );
    const ctrl = new AbortController();
    const c = collect();
    const p = api.chatStream(
      req,
      {
        ...c.handlers,
        onMeta: () => {
          ctrl.abort();
        },
      },
      ctrl.signal,
    );
    await expect(p).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('times out a stream that stalls between chunks', async () => {
    server.use(
      http.post(`${API}/api/chat/stream`, () => {
        const enc = new TextEncoder();
        return new HttpResponse(
          new ReadableStream({
            start(ctrl) {
              ctrl.enqueue(enc.encode(sse([['meta', metaFixture]])));
            },
          }),
        );
      }),
    );
    const e = await rejection(createApiClient(API, { timeoutMs: 50 }).chatStream(req, collect().handlers));
    expect(e.code).toBe('timeout');
  });

  it('accepts the full stream body in one chunk too', async () => {
    server.use(http.post(`${API}/api/chat/stream`, () => new HttpResponse(streamOk('ok'))));
    const c = collect();
    await api.chatStream(req, c.handlers);
    expect(c.tokens.join('')).toBe('ok');
  });
});

describe('api client: transcribe', () => {
  it('uploads multipart audio with the language hint', async () => {
    let fields: Record<string, unknown> = {};
    server.use(
      http.post(`${API}/api/transcribe`, async ({ request }) => {
        const form = await request.formData();
        const audio = form.get('audio');
        fields = {
          input_lang: form.get('input_lang'),
          // jsdom and Node each have their own File class, so duck-type it
          name: typeof audio === 'object' && audio ? audio.name : null,
          type: typeof audio === 'object' && audio ? audio.type : null,
        };
        return HttpResponse.json({ text: 'hello', input_lang: 'hi', duration_seconds: 1.5 });
      }),
    );
    const blob = new Blob([new Uint8Array([1, 2, 3])], { type: 'audio/webm;codecs=opus' });
    await expect(api.transcribe(blob, 'hi')).resolves.toEqual({
      text: 'hello',
      input_lang: 'hi',
      duration_seconds: 1.5,
    });
    // jsdom's FormData drops filenames when serialised by Node's fetch; the
    // backend keys on content type, which is what matters here.
    expect(fields).toMatchObject({ input_lang: 'hi', type: 'audio/webm;codecs=opus' });
  });
});
