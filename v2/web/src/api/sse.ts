/**
 * Minimal Server-Sent Events parser (https://html.spec.whatwg.org/multipage/server-sent-events.html).
 * EventSource cannot POST a JSON body, so the chat stream is read with fetch +
 * ReadableStream and parsed here. Handles CRLF/LF, multi-line `data:`, comments
 * and events split across network chunks.
 */
export type SseEvent = { event: string; data: string };

export function createSseParser(onEvent: (e: SseEvent) => void): {
  feed: (chunk: string) => void;
  end: () => void;
} {
  let buffer = '';
  let event = '';
  let data: string[] = [];

  const dispatch = () => {
    if (data.length > 0) onEvent({ event: event || 'message', data: data.join('\n') });
    event = '';
    data = [];
  };

  const processLine = (line: string) => {
    if (line === '') {
      dispatch();
      return;
    }
    if (line.startsWith(':')) return;
    const idx = line.indexOf(':');
    const field = idx === -1 ? line : line.slice(0, idx);
    let value = idx === -1 ? '' : line.slice(idx + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') event = value;
    else if (field === 'data') data.push(value);
  };

  return {
    feed(chunk: string) {
      buffer += chunk;
      const lines = buffer.split(/\r\n|\r|\n/);
      buffer = lines.pop() ?? '';
      for (const line of lines) processLine(line);
    },
    end() {
      if (buffer) processLine(buffer);
      buffer = '';
      dispatch();
    },
  };
}
