import { useState } from 'react';

import { citationAnchorId } from '../citations/anchors';
import { CitationList } from '../citations/CitationList';
import { AnswerText } from './AnswerText';
import type { AssistantMessage, Message } from './chat-state';

const PII_LABEL: Record<string, string> = {
  EMAIL: 'email address',
  PHONE: 'phone number',
  CREDIT_CARD: 'card number',
  AADHAAR: 'Aadhaar number',
  US_SSN: 'Social Security number',
  PAN: 'PAN',
  IBAN: 'bank account number',
  IP_ADDRESS: 'IP address',
};

function RedactionNotice({ m }: { m: AssistantMessage }) {
  if (m.redactions.length === 0) return null;
  const items = m.redactions.map(
    (r) => `${r.count} ${PII_LABEL[r.type] ?? r.type.toLowerCase()}${r.count > 1 ? 's' : ''}`,
  );
  return (
    <p className="mt-3 flex items-start gap-2 rounded-md bg-sky-50 px-3 py-2 text-xs text-sky-900">
      <span aria-hidden="true">🛡</span>
      <span>
        Personal data was masked before processing: {items.join(', ')}. It was not sent to the language model
        and is not logged.
      </span>
    </p>
  );
}

function Thinking() {
  return (
    <div role="status" className="flex items-center gap-2 text-sm text-stone-500">
      <span className="flex gap-1" aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-1.5 w-1.5 animate-bounce rounded-full bg-stone-400"
            style={{ animationDelay: `${i * 120}ms` }}
          />
        ))}
      </span>
      Searching the statutes…
    </div>
  );
}

function AssistantBody({ m, onRetry }: { m: AssistantMessage; onRetry: (id: string) => void }) {
  const [open, setOpen] = useState<ReadonlySet<string>>(new Set());
  const known = new Set(m.sources.map((s) => s.id));
  const toggle = (id: string) => {
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const cite = (id: string) => {
    setOpen((prev) => new Set(prev).add(id));
    requestAnimationFrame(() => {
      document
        .getElementById(citationAnchorId(m.id, id))
        ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  };

  if (m.status === 'pending') return <Thinking />;

  return (
    <>
      {m.grounded === false && m.status !== 'error' && (
        <p className="mb-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          No relevant passage was found in the indexed statutes, so this answer is not backed by a source.
        </p>
      )}
      {m.content && (
        <div lang={m.outputLang} className="font-serif text-[17px] leading-relaxed text-stone-900">
          <AnswerText text={m.content} knownIds={known} onCite={cite} />
          {m.status === 'streaming' && (
            <span
              aria-hidden="true"
              className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-stone-400 align-middle"
            />
          )}
        </div>
      )}
      {m.status === 'stopped' && <p className="mt-2 text-xs text-stone-500 italic">Stopped.</p>}
      {m.status === 'error' && m.error && (
        <div
          role="alert"
          className="mt-1 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-900"
        >
          <p>{m.error.message}</p>
          {m.error.retryAfterSeconds !== undefined && (
            <p className="mt-1 text-xs">You can try again in about {m.error.retryAfterSeconds} seconds.</p>
          )}
          <div className="mt-2 flex items-center gap-3">
            {m.error.retryable && (
              <button
                type="button"
                onClick={() => {
                  onRetry(m.id);
                }}
                className="rounded-md bg-red-900 px-3 py-1 text-xs font-medium text-white hover:bg-red-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-900"
              >
                Retry
              </button>
            )}
            {m.error.requestId && (
              <span className="font-mono text-[11px] text-red-700">ref {m.error.requestId.slice(0, 8)}</span>
            )}
          </div>
        </div>
      )}
      <RedactionNotice m={m} />
      {(m.status === 'done' || m.status === 'stopped' || m.status === 'streaming') && (
        <CitationList sources={m.sources} messageId={m.id} openIds={open} onToggle={toggle} />
      )}
    </>
  );
}

export function MessageItem({ message, onRetry }: { message: Message; onRetry: (id: string) => void }) {
  if (message.role === 'user') {
    return (
      <li className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-stone-900 px-4 py-2.5 text-[15px] text-stone-50 sm:max-w-[75%]">
          <span className="sr-only">You asked: </span>
          <p
            className="whitespace-pre-wrap"
            lang={message.inputLang === 'auto' ? undefined : message.inputLang}
          >
            {message.content}
          </p>
          {message.viaVoice && <p className="mt-1 text-[11px] text-stone-400">via voice</p>}
        </div>
      </li>
    );
  }
  return (
    <li>
      <article
        aria-label="LexiTalk answer"
        className="rounded-2xl border border-stone-200 bg-white px-4 py-4 shadow-sm sm:px-5"
      >
        <AssistantBody m={message} onRetry={onRetry} />
      </article>
    </li>
  );
}
