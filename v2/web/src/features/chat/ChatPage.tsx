import { useEffect, useRef, useState } from 'react';

import { useConfig, useHealth } from '../../app/queries';
import { LanguageControls } from '../language/LanguageControls';
import { useLanguagePrefs } from '../language/prefs';
import { Composer } from './Composer';
import { MessageItem } from './MessageItem';
import { useChat } from './useChat';

const EXAMPLES = [
  'What compensation does the President receive?',
  'How is a vacancy in the office of Vice President handled?',
  'What does Title 1 say about words importing the singular?',
  'राष्ट्रपति की शपथ के बारे में कानून क्या कहता है?',
];

function HealthBadge() {
  const health = useHealth();
  let label = 'Connecting…';
  let tone = 'bg-stone-300';
  if (health.isError) {
    label = 'Service unavailable';
    tone = 'bg-red-500';
  } else if (health.data?.status === 'degraded') {
    label = 'Index not ready';
    tone = 'bg-amber-500';
  } else if (health.data) {
    label = 'Online';
    tone = 'bg-emerald-500';
  }
  return (
    <span className="flex items-center gap-1.5 text-xs text-stone-500" role="status">
      <span className={`h-2 w-2 rounded-full ${tone}`} aria-hidden="true" />
      {label}
    </span>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mx-auto max-w-xl py-10 text-center sm:py-16">
      <h2 className="font-serif text-3xl text-stone-900 sm:text-4xl">Ask the statute.</h2>
      <p className="mt-3 text-[15px] leading-relaxed text-stone-600">
        LexiTalk answers questions about Titles 1–3 of the United States Code: general provisions, Congress
        and the Presidency. Every answer cites the passages it was drawn from. Type or speak in English or
        Hindi.
      </p>
      <ul className="mt-8 grid gap-2 text-left sm:grid-cols-2">
        {EXAMPLES.map((q) => (
          <li key={q}>
            <button
              type="button"
              onClick={() => {
                onPick(q);
              }}
              className="h-full w-full rounded-xl border border-stone-200 bg-white px-4 py-3 text-left text-sm text-stone-700 transition-colors hover:border-stone-400 hover:text-stone-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-600"
            >
              {q}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function ChatPage() {
  const config = useConfig();
  const chat = useChat();
  const [prefs, setPrefs] = useLanguagePrefs();
  const [prefill, setPrefill] = useState<{ text: string; nonce: number } | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const last = chat.messages.at(-1);
  const lastLen = last?.role === 'assistant' ? last.content.length : 0;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [chat.messages.length, lastLen]);

  const cfg = config.data;

  return (
    <div className="flex min-h-dvh flex-col">
      <header className="sticky top-0 z-10 border-b border-stone-200 bg-stone-50/90 backdrop-blur">
        <div className="mx-auto flex max-w-3xl flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5 sm:py-3">
          <div className="flex items-center gap-3">
            <h1 className="font-serif text-xl font-semibold tracking-tight">
              Lexi<span className="text-amber-700">Talk</span>
            </h1>
            <HealthBadge />
          </div>
          <div className="order-last w-full sm:order-none sm:ml-auto sm:w-auto">
            {cfg && (
              <LanguageControls
                prefs={prefs}
                onChange={setPrefs}
                inputOptions={cfg.input_languages}
                outputOptions={cfg.output_languages}
                disabled={chat.busy}
              />
            )}
          </div>
          <div className="ml-auto sm:ml-0">
            {chat.messages.length > 0 && (
              <button
                type="button"
                onClick={chat.reset}
                className="rounded-md px-2 py-1.5 text-xs font-medium text-stone-600 hover:bg-stone-200 hover:text-stone-900 focus-visible:outline-2 focus-visible:outline-amber-600"
              >
                New chat
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-3xl flex-1 px-4 pb-4">
        {config.isError && (
          <div
            role="alert"
            className="mt-6 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-900"
          >
            <p>Can’t reach the LexiTalk service. It may be starting up, which can take a minute.</p>
            <button
              type="button"
              onClick={() => void config.refetch()}
              className="mt-2 rounded-md bg-red-900 px-3 py-1 text-xs font-medium text-white hover:bg-red-800"
            >
              Try again
            </button>
          </div>
        )}
        {chat.messages.length === 0 ? (
          <EmptyState
            onPick={(q) => {
              setPrefill({ text: q, nonce: Date.now() });
            }}
          />
        ) : (
          <ol aria-label="Conversation" aria-live="polite" aria-busy={chat.busy} className="space-y-5 py-6">
            {chat.messages.map((m) => (
              <MessageItem key={m.id} message={m} onRetry={chat.retry} />
            ))}
          </ol>
        )}
        <div ref={endRef} />
      </main>

      <div className="sticky bottom-0 bg-gradient-to-t from-stone-50 via-stone-50 to-stone-50/0 pt-4">
        <div className="mx-auto max-w-3xl px-4 pb-3">
          <Composer
            key={prefill?.nonce ?? 0}
            busy={chat.busy}
            disabled={!cfg}
            maxChars={cfg?.max_message_chars ?? 2000}
            maxAudioSeconds={cfg?.max_audio_seconds ?? 60}
            inputLang={prefs.input}
            initialText={prefill?.text ?? ''}
            onStop={chat.stop}
            onSend={(text, viaVoice) => {
              chat.send(text, { inputLang: prefs.input, outputLang: prefs.output, viaVoice });
            }}
          />
          <p className="mt-2 text-center text-[11px] leading-snug text-stone-500">
            Legal information, not legal advice. Answers can be wrong — check the cited text. Personal data
            you type is masked before processing; your conversation stays in this browser tab.
          </p>
        </div>
      </div>
    </div>
  );
}
