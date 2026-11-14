import { createSseParser, type SseEvent } from './sse';

function parseAll(chunks: string[]): SseEvent[] {
  const out: SseEvent[] = [];
  const p = createSseParser((e) => out.push(e));
  chunks.forEach((c) => {
    p.feed(c);
  });
  p.end();
  return out;
}

describe('createSseParser', () => {
  it('parses named events', () => {
    expect(parseAll(['event: token\ndata: {"delta":"a"}\n\n'])).toEqual([
      { event: 'token', data: '{"delta":"a"}' },
    ]);
  });

  it('reassembles events split at arbitrary byte boundaries', () => {
    const body = 'event: meta\ndata: {"x":1}\n\nevent: token\ndata: {"delta":"hé"}\n\n';
    const oneByOne = body.split('');
    expect(parseAll(oneByOne)).toEqual(parseAll([body]));
    expect(parseAll(oneByOne)).toHaveLength(2);
  });

  it('handles CRLF, comments, multi-line data and default event name', () => {
    const events = parseAll([': keep-alive\r\ndata: line1\r\ndata: line2\r\n\r\n']);
    expect(events).toEqual([{ event: 'message', data: 'line1\nline2' }]);
  });

  it('flushes a final event without a trailing blank line', () => {
    expect(parseAll(['event: done\ndata: {}'])).toEqual([{ event: 'done', data: '{}' }]);
  });
});
