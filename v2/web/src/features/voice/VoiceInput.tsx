import { useMutation } from '@tanstack/react-query';
import { useEffect, useState } from 'react';

import { ApiError } from '../../api/client';
import type { InputLang } from '../../api/schemas';
import { useApi } from '../../app/api-context';
import { isRecordingSupported, type RecorderState, useRecorder } from './useRecorder';

const ERROR_TEXT: Record<Extract<RecorderState, { kind: 'error' }>['reason'], string> = {
  unsupported: 'Voice input is not supported in this browser.',
  denied: 'Microphone access was blocked. Allow it in your browser settings to use voice.',
  'no-device': 'No microphone was found.',
  failed: 'Recording failed. Please try again.',
};

function Elapsed({ startedAt, max }: { startedAt: number; max: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => {
      setNow(Date.now());
    }, 250);
    return () => {
      clearInterval(t);
    };
  }, []);
  const s = Math.min(max, Math.floor((now - startedAt) / 1000));
  return (
    <span className="tabular-nums" aria-hidden="true">
      0:{String(s).padStart(2, '0')} / 0:{String(max).padStart(2, '0')}
    </span>
  );
}

/**
 * Mic button → record → POST /api/transcribe → hand the transcript to the
 * composer for review. Nothing is sent to the chat until the user presses Send.
 */
export function VoiceInput({
  inputLang,
  maxSeconds,
  disabled,
  onTranscript,
}: {
  inputLang: InputLang;
  maxSeconds: number;
  disabled: boolean;
  onTranscript: (text: string) => void;
}) {
  const api = useApi();
  const transcribe = useMutation({
    mutationFn: (blob: Blob) => api.transcribe(blob, inputLang),
    onSuccess: (r) => {
      onTranscript(r.text);
    },
  });
  const rec = useRecorder({
    maxSeconds,
    onComplete: (blob) => {
      transcribe.mutate(blob);
    },
  });

  const recording = rec.state.kind === 'recording';
  const busy = rec.state.kind === 'requesting' || transcribe.isPending;
  const supported = isRecordingSupported();

  let status: string | null = null;
  if (rec.state.kind === 'error') status = ERROR_TEXT[rec.state.reason];
  else if (transcribe.isError) {
    status = transcribe.error instanceof ApiError ? transcribe.error.message : 'Transcription failed.';
  } else if (transcribe.isSuccess && !transcribe.data.text) status = 'No speech was detected. Try again.';

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        disabled={disabled || busy || !supported}
        onClick={() => {
          transcribe.reset();
          rec.dismissError();
          if (recording) rec.stop();
          else void rec.start();
        }}
        aria-pressed={recording}
        aria-label={recording ? 'Stop recording' : 'Record a question'}
        title={supported ? undefined : ERROR_TEXT.unsupported}
        className={`grid h-10 w-10 shrink-0 place-items-center rounded-full border transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-600 disabled:cursor-not-allowed disabled:opacity-40 ${
          recording
            ? 'border-red-600 bg-red-600 text-white hover:bg-red-700'
            : 'border-stone-300 bg-white text-stone-700 hover:bg-stone-100'
        }`}
      >
        {busy ? (
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-stone-300 border-t-stone-700" />
        ) : recording ? (
          <span className="h-3.5 w-3.5 rounded-sm bg-white" />
        ) : (
          <svg
            aria-hidden="true"
            viewBox="0 0 24 24"
            className="h-5 w-5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
          >
            <rect x="9" y="3" width="6" height="11" rx="3" />
            <path d="M5 11a7 7 0 0 0 14 0M12 18v3" strokeLinecap="round" />
          </svg>
        )}
      </button>
      <span role="status" aria-live="polite" className="text-xs text-stone-600">
        {recording && rec.state.kind === 'recording' && (
          <span className="flex items-center gap-2 text-red-700">
            <span className="h-2 w-2 animate-pulse rounded-full bg-red-600" aria-hidden="true" />
            <span>Recording…</span>
            <Elapsed startedAt={rec.state.startedAt} max={maxSeconds} />
          </span>
        )}
        {rec.state.kind === 'requesting' && 'Waiting for microphone permission…'}
        {transcribe.isPending && 'Transcribing…'}
        {status && <span className="text-red-700">{status}</span>}
      </span>
    </div>
  );
}
