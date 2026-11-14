import type { HistoryTurn, InputLang, OutputLang, Redaction, Source, StreamMeta } from '../../api/schemas';

export type ChatError = {
  code: string;
  message: string;
  retryable: boolean;
  retryAfterSeconds?: number | undefined;
  requestId?: string | undefined;
};

export type UserMessage = {
  id: string;
  role: 'user';
  content: string;
  inputLang: InputLang;
  outputLang: OutputLang;
  viaVoice: boolean;
};

export type AssistantStatus = 'pending' | 'streaming' | 'done' | 'stopped' | 'error';

export type AssistantMessage = {
  id: string;
  role: 'assistant';
  replyTo: string;
  status: AssistantStatus;
  content: string;
  outputLang: OutputLang;
  sources: Source[];
  grounded: boolean | null;
  detectedLang: 'en' | 'hi' | null;
  redactions: Redaction[];
  requestId: string | null;
  error: ChatError | null;
};

export type Message = UserMessage | AssistantMessage;
export type ChatState = { messages: Message[] };

export type ChatAction =
  | { type: 'send'; user: UserMessage; assistantId: string }
  | { type: 'retry'; assistantId: string }
  | { type: 'meta'; assistantId: string; meta: StreamMeta }
  | { type: 'token'; assistantId: string; delta: string }
  | { type: 'done'; assistantId: string; redactions: Redaction[] }
  | {
      type: 'complete';
      assistantId: string;
      answer: string;
      meta: StreamMeta;
      redactions: Redaction[];
    }
  | { type: 'stopped'; assistantId: string }
  | { type: 'error'; assistantId: string; error: ChatError }
  | { type: 'reset' };

export const initialChatState: ChatState = { messages: [] };

function updateAssistant(
  state: ChatState,
  id: string,
  fn: (m: AssistantMessage) => AssistantMessage,
): ChatState {
  return {
    messages: state.messages.map((m) => (m.role === 'assistant' && m.id === id ? fn(m) : m)),
  };
}

function applyMeta(m: AssistantMessage, meta: StreamMeta): AssistantMessage {
  return {
    ...m,
    sources: meta.sources,
    grounded: meta.grounded,
    detectedLang: meta.detected_lang,
    requestId: meta.request_id,
  };
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'send':
      return {
        messages: [
          ...state.messages,
          action.user,
          {
            id: action.assistantId,
            role: 'assistant',
            replyTo: action.user.id,
            status: 'pending',
            content: '',
            outputLang: action.user.outputLang,
            sources: [],
            grounded: null,
            detectedLang: null,
            redactions: [],
            requestId: null,
            error: null,
          },
        ],
      };
    case 'retry':
      return updateAssistant(state, action.assistantId, (m) => ({
        ...m,
        status: 'pending',
        content: '',
        sources: [],
        grounded: null,
        redactions: [],
        error: null,
      }));
    case 'meta':
      return updateAssistant(state, action.assistantId, (m) => ({
        ...applyMeta(m, action.meta),
        status: 'streaming',
      }));
    case 'token':
      return updateAssistant(state, action.assistantId, (m) => ({
        ...m,
        status: 'streaming',
        content: m.content + action.delta,
      }));
    case 'done':
      return updateAssistant(state, action.assistantId, (m) => ({
        ...m,
        status: 'done',
        redactions: action.redactions,
      }));
    case 'complete':
      return updateAssistant(state, action.assistantId, (m) => ({
        ...applyMeta(m, action.meta),
        status: 'done',
        content: action.answer,
        redactions: action.redactions,
      }));
    case 'stopped':
      return updateAssistant(state, action.assistantId, (m) => ({ ...m, status: 'stopped' }));
    case 'error':
      return updateAssistant(state, action.assistantId, (m) => ({
        ...m,
        status: 'error',
        error: action.error,
      }));
    case 'reset':
      return initialChatState;
  }
}

/** Completed turns sent as conversation context (the server masks and caps them). */
export function buildHistory(messages: Message[], maxTurns = 6): HistoryTurn[] {
  const turns: HistoryTurn[] = [];
  for (const m of messages) {
    if (m.role === 'user') {
      const reply = messages.find((x) => x.role === 'assistant' && x.replyTo === m.id);
      if (reply?.role === 'assistant' && reply.status === 'done' && reply.content) {
        turns.push({ role: 'user', content: m.content }, { role: 'assistant', content: reply.content });
      }
    }
  }
  return turns.slice(-maxTurns);
}

export function isBusy(state: ChatState): boolean {
  return state.messages.some(
    (m) => m.role === 'assistant' && (m.status === 'pending' || m.status === 'streaming'),
  );
}
