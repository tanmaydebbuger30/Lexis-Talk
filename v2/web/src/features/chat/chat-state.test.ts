import { metaFixture } from '../../test/fixtures';
import {
  buildHistory,
  chatReducer,
  type ChatState,
  initialChatState,
  isBusy,
  type UserMessage,
} from './chat-state';

const user = (id: string, content = 'q'): UserMessage => ({
  id,
  role: 'user',
  content,
  inputLang: 'auto',
  outputLang: 'en',
  viaVoice: false,
});

function run(...actions: Parameters<typeof chatReducer>[1][]): ChatState {
  return actions.reduce(chatReducer, initialChatState);
}

describe('chatReducer', () => {
  it('streams meta → tokens → done into one assistant message', () => {
    const s = run(
      { type: 'send', user: user('u1'), assistantId: 'a1' },
      { type: 'meta', assistantId: 'a1', meta: metaFixture },
      { type: 'token', assistantId: 'a1', delta: 'Hel' },
      { type: 'token', assistantId: 'a1', delta: 'lo' },
      { type: 'done', assistantId: 'a1', redactions: [{ type: 'EMAIL', count: 1 }] },
    );
    const a = s.messages[1];
    expect(a).toMatchObject({ role: 'assistant', status: 'done', content: 'Hello', grounded: true });
    expect(a?.role === 'assistant' && a.sources).toHaveLength(2);
    expect(isBusy(s)).toBe(false);
  });

  it('retry clears the previous attempt', () => {
    const s = run(
      { type: 'send', user: user('u1'), assistantId: 'a1' },
      { type: 'token', assistantId: 'a1', delta: 'partial' },
      { type: 'error', assistantId: 'a1', error: { code: 'x', message: 'm', retryable: true } },
      { type: 'retry', assistantId: 'a1' },
    );
    expect(s.messages[1]).toMatchObject({ status: 'pending', content: '', error: null });
    expect(isBusy(s)).toBe(true);
  });
});

describe('buildHistory', () => {
  it('includes only completed exchanges and caps the length', () => {
    let s = initialChatState;
    for (let i = 0; i < 5; i++) {
      s = chatReducer(s, { type: 'send', user: user(`u${i}`, `q${i}`), assistantId: `a${i}` });
      s = chatReducer(s, { type: 'token', assistantId: `a${i}`, delta: `ans${i}` });
      if (i !== 2) s = chatReducer(s, { type: 'done', assistantId: `a${i}`, redactions: [] });
      else
        s = chatReducer(s, {
          type: 'error',
          assistantId: `a${i}`,
          error: { code: 'e', message: 'e', retryable: true },
        });
    }
    const h = buildHistory(s.messages, 4);
    expect(h).toEqual([
      { role: 'user', content: 'q3' },
      { role: 'assistant', content: 'ans3' },
      { role: 'user', content: 'q4' },
      { role: 'assistant', content: 'ans4' },
    ]);
    expect(buildHistory(s.messages, 100).some((t) => t.content === 'q2')).toBe(false);
  });
});
