import { useCallback, useEffect, useLayoutEffect, useReducer, useRef } from 'react';
import { z } from 'zod';

import { ApiError } from '../../api/client';
import type { InputLang, OutputLang, StreamMeta } from '../../api/schemas';
import { useApi } from '../../app/api-context';
import {
  type AssistantMessage,
  buildHistory,
  type ChatError,
  chatReducer,
  type ChatState,
  initialChatState,
  isBusy,
  type Message,
  type UserMessage,
} from './chat-state';

const STORAGE_KEY = 'lexitalk.chat.v1';

/**
 * Conversation lives in sessionStorage only: it never leaves this browser tab
 * except as the request it was typed for, and is gone when the tab closes.
 * Validated on load so a stale/corrupt entry can't crash the app.
 */
const PersistedSchema = z.object({
  messages: z.array(z.looseObject({ id: z.string(), role: z.enum(['user', 'assistant']) })),
});

function loadState(): ChatState {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return initialChatState;
    const parsed = PersistedSchema.safeParse(JSON.parse(raw));
    if (!parsed.success) return initialChatState;
    // In-flight answers cannot resume after a reload.
    const messages = (parsed.data.messages as unknown as Message[]).map((m) =>
      m.role === 'assistant' && (m.status === 'pending' || m.status === 'streaming')
        ? { ...m, status: 'stopped' as const }
        : m,
    );
    return { messages };
  } catch {
    return initialChatState;
  }
}

function toChatError(e: unknown): ChatError {
  if (e instanceof ApiError) {
    return {
      code: e.code,
      message: e.message,
      retryable: e.retryable,
      retryAfterSeconds: e.retryAfterSeconds,
      requestId: e.requestId,
    };
  }
  return { code: 'unknown', message: 'Something went wrong.', retryable: true };
}

const newId = () => crypto.randomUUID();

export type SendOptions = { inputLang: InputLang; outputLang: OutputLang; viaVoice?: boolean };

export function useChat() {
  const api = useApi();
  const [state, dispatch] = useReducer(chatReducer, undefined, loadState);
  const abortRef = useRef<AbortController | null>(null);
  // Latest state for event handlers (send/retry) without re-creating them.
  const stateRef = useRef(state);
  useLayoutEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    try {
      if (state.messages.length === 0) sessionStorage.removeItem(STORAGE_KEY);
      else sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      /* storage full or disabled: the chat still works, it just won't survive a reload */
    }
  }, [state]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const run = useCallback(
    async (user: UserMessage, assistantId: string, priorMessages: Message[]) => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      const req = {
        message: user.content,
        input_lang: user.inputLang,
        output_lang: user.outputLang,
        history: buildHistory(priorMessages),
      };
      try {
        try {
          await api.chatStream(
            req,
            {
              onMeta: (meta) => {
                dispatch({ type: 'meta', assistantId, meta });
              },
              onToken: (delta) => {
                dispatch({ type: 'token', assistantId, delta });
              },
              onDone: (redactions) => {
                dispatch({ type: 'done', assistantId, redactions });
              },
            },
            ctrl.signal,
          );
        } catch (e) {
          // Some proxies/browsers can't stream: fall back to the JSON endpoint.
          if (!(e instanceof ApiError && e.code === 'stream_unsupported')) throw e;
          const r = await api.chat(req, ctrl.signal);
          const meta: StreamMeta = {
            request_id: r.request_id,
            detected_lang: r.detected_lang,
            query_used_for_retrieval: r.query_used_for_retrieval,
            grounded: r.grounded,
            sources: r.sources,
          };
          dispatch({ type: 'complete', assistantId, answer: r.answer, meta, redactions: r.redactions });
        }
      } catch (e) {
        if (ctrl.signal.aborted) {
          dispatch({ type: 'stopped', assistantId });
          return;
        }
        dispatch({ type: 'error', assistantId, error: toChatError(e) });
      } finally {
        if (abortRef.current === ctrl) abortRef.current = null;
      }
    },
    [api],
  );

  const send = useCallback(
    (text: string, opts: SendOptions) => {
      const content = text.trim();
      if (!content || isBusy(stateRef.current)) return;
      const prior = stateRef.current.messages;
      const user: UserMessage = {
        id: newId(),
        role: 'user',
        content,
        inputLang: opts.inputLang,
        outputLang: opts.outputLang,
        viaVoice: opts.viaVoice ?? false,
      };
      const assistantId = newId();
      dispatch({ type: 'send', user, assistantId });
      void run(user, assistantId, prior);
    },
    [run],
  );

  const retry = useCallback(
    (assistantId: string) => {
      const msgs = stateRef.current.messages;
      const idx = msgs.findIndex((m) => m.id === assistantId);
      const a = msgs[idx] as AssistantMessage | undefined;
      const user = msgs.find((m): m is UserMessage => m.role === 'user' && m.id === a?.replyTo);
      if (!a || !user || isBusy(stateRef.current)) return;
      dispatch({ type: 'retry', assistantId });
      void run(user, assistantId, msgs.slice(0, msgs.indexOf(user)));
    },
    [run],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    dispatch({ type: 'reset' });
  }, []);

  return { messages: state.messages, busy: isBusy(state), send, retry, stop, reset };
}
