import { screen, waitFor, within } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

import type { ChatRequest } from '../../api/schemas';
import { API, chunkedStream, errorBody, metaFixture, sse, streamOk } from '../../test/fixtures';
import { renderApp } from '../../test/render';
import { server } from '../../test/server';

const composer = () => screen.getByRole('textbox', { name: /ask a question/i });

async function ask(user: ReturnType<typeof renderApp>['user'], text: string) {
  await waitFor(() => {
    expect(composer()).toBeEnabled();
  });
  await user.type(composer(), text);
  await user.keyboard('{Enter}');
}

function captureRequests() {
  const bodies: ChatRequest[] = [];
  server.use(
    http.post(`${API}/api/chat/stream`, async ({ request }) => {
      bodies.push((await request.json()) as ChatRequest);
      return new HttpResponse(chunkedStream(streamOk()), {
        headers: { 'Content-Type': 'text/event-stream' },
      });
    }),
  );
  return bodies;
}

describe('chat flow', () => {
  it('shows the empty state, then streams an answer with citations', async () => {
    const { user } = renderApp();
    expect(screen.getByRole('heading', { name: 'Ask the statute.' })).toBeInTheDocument();
    expect(await screen.findByText('Online')).toBeInTheDocument();

    await ask(user, 'What is the President paid?');

    const answer = await screen.findByRole('article', { name: 'LexiTalk answer' });
    await within(answer).findByText(/receives \$400,000 a year/);
    expect(screen.getByText('What is the President paid?')).toBeInTheDocument();
    expect(composer()).toHaveValue('');

    // only the source used in the prompt is listed by default
    const sources = within(answer).getByRole('region', { name: 'Sources' });
    expect(within(sources).getByText('Sources (1 used)')).toBeInTheDocument();
    const s1 = within(sources).getByRole('button', { name: /U\.S\. Code Title 3/ });
    expect(s1).toHaveAttribute('aria-expanded', 'false');

    // clicking the inline [S1] marker opens that source
    await user.click(within(answer).getByRole('button', { name: 'Show source S1' }));
    expect(s1).toHaveAttribute('aria-expanded', 'true');
    expect(within(sources).getByText(/aggregate amount of \$400,000/)).toBeVisible();

    // unused retrieved passages are available on demand
    await user.click(within(sources).getByRole('button', { name: /1 retrieved passage not used/ }));
    expect(within(sources).getByText('Retrieved, not used')).toBeInTheDocument();
  });

  it('sends language choices and prior turns as history', async () => {
    const bodies = captureRequests();
    const { user } = renderApp();
    await screen.findByRole('combobox', { name: 'Reply in' });
    await user.selectOptions(screen.getByRole('combobox', { name: 'Reply in' }), 'hi');
    await user.selectOptions(screen.getByRole('combobox', { name: 'Ask in' }), 'en');

    await ask(user, 'first question');
    await screen.findByText(/receives \$400,000 a year/);
    await ask(user, 'follow up');
    await waitFor(() => {
      expect(bodies).toHaveLength(2);
    });

    expect(bodies[0]).toEqual({
      message: 'first question',
      input_lang: 'en',
      output_lang: 'hi',
      history: [],
    });
    expect(bodies[1]?.history).toEqual([
      { role: 'user', content: 'first question' },
      { role: 'assistant', content: 'The President receives $400,000 a year [S1].' },
    ]);
    // preference persists; conversation content is not written to localStorage
    expect(localStorage.getItem('lexitalk.lang.v1')).toBe('{"input":"en","output":"hi"}');
    expect(JSON.stringify(localStorage)).not.toContain('first question');
  });

  it('shows the PII redaction notice reported by the server', async () => {
    server.use(
      http.post(
        `${API}/api/chat/stream`,
        () =>
          new HttpResponse(
            sse([
              ['meta', metaFixture],
              ['token', { delta: 'Answer [S1].' }],
              [
                'done',
                {
                  redactions: [
                    { type: 'EMAIL', count: 1 },
                    { type: 'PHONE', count: 2 },
                  ],
                },
              ],
            ]),
          ),
      ),
    );
    const { user } = renderApp();
    await ask(user, 'I am a@b.io, call 410-555-0199');
    expect(
      await screen.findByText(/masked before processing: 1 email address, 2 phone numbers/),
    ).toBeInTheDocument();
  });

  it('warns when the answer is not grounded in any source', async () => {
    server.use(
      http.post(
        `${API}/api/chat/stream`,
        () =>
          new HttpResponse(
            sse([
              ['meta', { ...metaFixture, grounded: false, sources: [] }],
              ['token', { delta: 'The indexed documents do not cover this.' }],
              ['done', { redactions: [] }],
            ]),
          ),
      ),
    );
    const { user } = renderApp();
    await ask(user, 'zebra migration');
    expect(await screen.findByText(/not backed by a source/)).toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Sources' })).not.toBeInTheDocument();
  });

  it('shows a retryable error and recovers on Retry', async () => {
    let calls = 0;
    server.use(
      http.post(`${API}/api/chat/stream`, () => {
        calls += 1;
        if (calls === 1) {
          return HttpResponse.json(errorBody('rate_limited', 'Too many requests. Please wait a moment.'), {
            status: 429,
            headers: { 'Retry-After': '20' },
          });
        }
        return new HttpResponse(streamOk());
      }),
    );
    const { user } = renderApp();
    await ask(user, 'q');
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Too many requests');
    expect(alert).toHaveTextContent('about 20 seconds');
    expect(alert).toHaveTextContent('ref req-err0');

    await user.click(within(alert).getByRole('button', { name: 'Retry' }));
    expect(await screen.findByText(/receives \$400,000 a year/)).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(calls).toBe(2);
  });

  it('does not offer Retry for validation errors', async () => {
    server.use(
      http.post(`${API}/api/chat/stream`, () =>
        HttpResponse.json(errorBody('message_too_long', 'Messages are limited to 2000 characters.'), {
          status: 422,
        }),
      ),
    );
    const { user } = renderApp();
    await ask(user, 'q');
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('limited to 2000 characters');
    expect(within(alert).queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
  });

  it('can stop a streaming answer and keeps the partial text', async () => {
    server.use(
      http.post(`${API}/api/chat/stream`, () => {
        const enc = new TextEncoder();
        return new HttpResponse(
          new ReadableStream({
            start(c) {
              c.enqueue(
                enc.encode(
                  sse([
                    ['meta', metaFixture],
                    ['token', { delta: 'Partial answer' }],
                  ]),
                ),
              );
            },
          }),
        );
      }),
    );
    const { user } = renderApp();
    await ask(user, 'q');
    await screen.findByText(/Partial answer/);
    await user.click(screen.getByRole('button', { name: 'Stop' }));
    expect(await screen.findByText('Stopped.')).toBeInTheDocument();
    expect(screen.getByText(/Partial answer/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Send' })).toBeInTheDocument();
  });

  it('blocks over-long messages client-side', async () => {
    server.use(
      http.get(`${API}/api/config`, () =>
        HttpResponse.json({
          input_languages: ['auto', 'en', 'hi'],
          output_languages: ['en', 'hi'],
          tts_enabled: false,
          max_message_chars: 10,
          max_audio_seconds: 60,
          max_audio_bytes: 1000,
        }),
      ),
    );
    const { user } = renderApp();
    await waitFor(() => {
      expect(composer()).toBeEnabled();
    });
    await user.type(composer(), 'this is far too long');
    expect(screen.getByText('10 characters over the limit')).toBeInTheDocument();
    expect(composer()).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled();
  });

  it('keeps the conversation in sessionStorage and clears it on New chat', async () => {
    const first = renderApp();
    await ask(first.user, 'remember me');
    await screen.findByText(/receives \$400,000 a year/);
    first.unmount();

    const second = renderApp();
    expect(await screen.findByText('remember me')).toBeInTheDocument();
    await second.user.click(screen.getByRole('button', { name: 'New chat' }));
    expect(screen.getByRole('heading', { name: 'Ask the statute.' })).toBeInTheDocument();
    expect(sessionStorage.getItem('lexitalk.chat.v1')).toBeNull();
  });

  it('ignores corrupt persisted state', async () => {
    sessionStorage.setItem('lexitalk.chat.v1', '{"messages":"nope"}');
    renderApp();
    expect(await screen.findByRole('heading', { name: 'Ask the statute.' })).toBeInTheDocument();
  });

  it('example questions prefill the composer without sending', async () => {
    let sent = false;
    server.use(
      http.post(`${API}/api/chat/stream`, () => {
        sent = true;
        return new HttpResponse(streamOk());
      }),
    );
    const { user } = renderApp();
    await waitFor(() => {
      expect(composer()).toBeEnabled();
    });
    await user.click(screen.getByRole('button', { name: /What compensation does the President receive/ }));
    expect(composer()).toHaveValue('What compensation does the President receive?');
    expect(sent).toBe(false);
  });

  it('shows a service error when config cannot load', async () => {
    server.use(http.get(`${API}/api/config`, () => HttpResponse.error()));
    renderApp();
    expect(
      await screen.findByText(/Can’t reach the LexiTalk service/, {}, { timeout: 5000 }),
    ).toBeInTheDocument();
    expect(composer()).toBeDisabled();
  });
});

describe('answer rendering safety', () => {
  it('renders model output as text, never HTML', async () => {
    server.use(
      http.post(
        `${API}/api/chat/stream`,
        () =>
          new HttpResponse(
            sse([
              ['meta', metaFixture],
              ['token', { delta: '<img src=x onerror="alert(1)"> [S1] [S9]' }],
              ['done', { redactions: [] }],
            ]),
          ),
      ),
    );
    const { user, container } = renderApp();
    await ask(user, 'q');
    const answer = await screen.findByRole('article');
    await within(answer).findByText(/<img src=x onerror="alert\(1\)">/);
    expect(container.querySelector('img')).toBeNull();
    // unknown citation ids stay as plain text
    expect(within(answer).queryByRole('button', { name: 'Show source S9' })).not.toBeInTheDocument();
    expect(within(answer).getByText(/\[S9\]/)).toBeInTheDocument();
  });
});
