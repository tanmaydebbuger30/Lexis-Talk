import { type KeyboardEvent, useEffect, useId, useRef, useState } from 'react';

import type { InputLang } from '../../api/schemas';
import { VoiceInput } from '../voice/VoiceInput';

export function Composer({
  busy,
  disabled,
  maxChars,
  maxAudioSeconds,
  inputLang,
  onSend,
  onStop,
  initialText,
}: {
  /** An answer is in flight: show Stop instead of Send. */
  busy: boolean;
  /** The service isn't ready (config not loaded): nothing can be typed or sent. */
  disabled: boolean;
  maxChars: number;
  maxAudioSeconds: number;
  inputLang: InputLang;
  onSend: (text: string, viaVoice: boolean) => void;
  onStop: () => void;
  /** Prefilled text (e.g. an example question). Remount with a new `key` to replace it. */
  initialText: string;
}) {
  const [text, setText] = useState(initialText);
  const [fromVoice, setFromVoice] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);
  const hintId = useId();
  const tooLong = text.length > maxChars;
  const canSend = !busy && !disabled && text.trim().length > 0 && !tooLong;

  useEffect(() => {
    if (initialText) ref.current?.focus();
  }, [initialText]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);

  const submit = (e?: { preventDefault: () => void }) => {
    e?.preventDefault();
    if (!canSend) return;
    onSend(text, fromVoice);
    setText('');
    setFromVoice(false);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form
      onSubmit={submit}
      className="rounded-2xl border border-stone-300 bg-white p-2 shadow-sm focus-within:border-stone-500"
    >
      {fromVoice && (
        <p className="px-2 pt-1 text-xs text-amber-800">
          Transcribed from your recording. Check it, edit if needed, then send.
        </p>
      )}
      <label htmlFor="composer" className="sr-only">
        Ask a question about U.S. federal statutes
      </label>
      <textarea
        id="composer"
        ref={ref}
        rows={1}
        value={text}
        onChange={(e) => {
          setText(e.target.value);
        }}
        onKeyDown={onKeyDown}
        placeholder="Ask about Titles 1–3 of the U.S. Code…"
        aria-describedby={hintId}
        aria-invalid={tooLong}
        disabled={disabled}
        className="block max-h-[200px] w-full resize-none bg-transparent disabled:cursor-not-allowed px-2 py-2 text-[15px] text-stone-900 placeholder:text-stone-400 focus:outline-none"
      />
      <div className="flex items-center justify-between gap-2 px-1 pt-1">
        <VoiceInput
          inputLang={inputLang}
          maxSeconds={maxAudioSeconds}
          disabled={busy || disabled}
          onTranscript={(t) => {
            setText(t);
            setFromVoice(t.length > 0);
            ref.current?.focus();
          }}
        />
        <div className="flex items-center gap-3">
          <span
            id={hintId}
            className={`text-xs tabular-nums ${tooLong ? 'font-medium text-red-700' : 'text-stone-400'}`}
          >
            {tooLong ? `${text.length - maxChars} characters over the limit` : `${text.length}/${maxChars}`}
          </span>
          {busy ? (
            <button
              type="button"
              onClick={onStop}
              className="rounded-full border border-stone-300 bg-white px-4 py-2 text-sm font-medium text-stone-800 hover:bg-stone-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-600"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!canSend}
              className="rounded-full bg-stone-900 px-4 py-2 text-sm font-medium text-white hover:bg-stone-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-600 disabled:cursor-not-allowed disabled:bg-stone-300"
            >
              Send
            </button>
          )}
        </div>
      </div>
    </form>
  );
}
