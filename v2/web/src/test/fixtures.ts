import { http, HttpResponse } from 'msw';

import type { ChatResponse, ConfigResponse, HealthResponse, Source, StreamMeta } from '../api/schemas';

export const API = 'http://api.test';

export const sourceFixture = (over: Partial<Source> = {}): Source => ({
  id: 'S1',
  document_id: 'usc-title-03',
  title: 'U.S. Code Title 3 — The President',
  page: 4,
  snippet:
    'The President shall receive in full for his services during the term for which he shall have been elected compensation in the aggregate amount of $400,000 a year.',
  score: 0.82,
  rank: 1,
  used_in_prompt: true,
  ...over,
});

export const configFixture: ConfigResponse = {
  input_languages: ['auto', 'en', 'hi'],
  output_languages: ['en', 'hi'],
  tts_enabled: false,
  max_message_chars: 2000,
  max_audio_seconds: 60,
  max_audio_bytes: 5 * 1024 * 1024,
};

export const healthFixture: HealthResponse = {
  status: 'ok',
  index_ready: true,
  chunks_indexed: 4213,
  version: '0.1.0',
};

export const metaFixture: StreamMeta = {
  request_id: 'req-1234abcd',
  detected_lang: 'en',
  query_used_for_retrieval: 'What is the President paid?',
  grounded: true,
  sources: [
    sourceFixture(),
    sourceFixture({ id: 'S2', rank: 2, score: 0.41, page: 7, used_in_prompt: false }),
  ],
};

export const chatFixture: ChatResponse = {
  answer: 'The President receives $400,000 a year [S1].',
  detected_lang: 'en',
  query_used_for_retrieval: 'What is the President paid?',
  grounded: true,
  sources: metaFixture.sources,
  redactions: [],
  request_id: 'req-1234abcd',
};

export function sse(events: [string, unknown][]): string {
  return events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join('');
}

/** A streaming body that emits the given string in small, awkward pieces. */
export function chunkedStream(body: string, size = 7): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(body);
  let i = 0;
  return new ReadableStream({
    pull(ctrl) {
      if (i >= bytes.length) {
        ctrl.close();
        return;
      }
      ctrl.enqueue(bytes.slice(i, i + size));
      i += size;
    },
  });
}

export const streamOk = (answer = 'The President receives $400,000 a year [S1].') =>
  sse([
    ['meta', metaFixture],
    ...answer.match(/.{1,6}/gs)!.map((d): [string, unknown] => ['token', { delta: d }]),
    ['done', { redactions: [] }],
  ]);

export const errorBody = (code: string, message: string) => ({
  error: { code, message, request_id: 'req-err00001' },
});

export const defaultHandlers = [
  http.get(`${API}/health`, () => HttpResponse.json(healthFixture)),
  http.get(`${API}/api/config`, () => HttpResponse.json(configFixture)),
  http.post(`${API}/api/chat`, () => HttpResponse.json(chatFixture)),
  http.post(
    `${API}/api/chat/stream`,
    () => new HttpResponse(chunkedStream(streamOk()), { headers: { 'Content-Type': 'text/event-stream' } }),
  ),
  http.post(`${API}/api/transcribe`, () =>
    HttpResponse.json({
      text: 'what does title 3 say about the president',
      input_lang: 'en',
      duration_seconds: 2.4,
    }),
  ),
];
