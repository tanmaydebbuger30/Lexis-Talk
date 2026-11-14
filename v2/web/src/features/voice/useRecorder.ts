import { useCallback, useEffect, useRef, useState } from 'react';

export type RecorderState =
  | { kind: 'idle' }
  | { kind: 'requesting' }
  | { kind: 'recording'; startedAt: number }
  | { kind: 'error'; reason: 'unsupported' | 'denied' | 'no-device' | 'failed' };

// Whisper on the server decodes all of these via ffmpeg; prefer Opus for size.
const MIME_CANDIDATES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];

export function pickMimeType(isSupported: (t: string) => boolean): string | undefined {
  return MIME_CANDIDATES.find((t) => isSupported(t));
}

export function isRecordingSupported(): boolean {
  // mediaDevices is undefined on insecure (non-HTTPS) origins despite the DOM typings.
  return (
    typeof window !== 'undefined' &&
    'MediaRecorder' in window &&
    'mediaDevices' in navigator &&
    typeof navigator.mediaDevices.getUserMedia === 'function'
  );
}

/**
 * Records microphone audio into a single Blob. Audio stays in memory: it is
 * never written to storage, and the mic track is released as soon as
 * recording stops so the browser's recording indicator turns off.
 */
export function useRecorder({
  maxSeconds,
  onComplete,
}: {
  maxSeconds: number;
  onComplete: (b: Blob) => void;
}) {
  const [state, setState] = useState<RecorderState>({ kind: 'idle' });
  const recRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelledRef = useRef(false);
  const onCompleteRef = useRef(onComplete);
  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  const release = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    streamRef.current?.getTracks().forEach((t) => {
      t.stop();
    });
    streamRef.current = null;
    recRef.current = null;
  }, []);

  useEffect(
    () => () => {
      cancelledRef.current = true;
      if (recRef.current?.state === 'recording') recRef.current.stop();
      release();
    },
    [release],
  );

  const start = useCallback(async () => {
    if (!isRecordingSupported()) {
      setState({ kind: 'error', reason: 'unsupported' });
      return;
    }
    setState({ kind: 'requesting' });
    cancelledRef.current = false;
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch (e) {
      const name = e instanceof DOMException ? e.name : '';
      setState({
        kind: 'error',
        reason: name === 'NotAllowedError' ? 'denied' : name === 'NotFoundError' ? 'no-device' : 'failed',
      });
      return;
    }
    streamRef.current = stream;
    const mimeType = pickMimeType((t) => MediaRecorder.isTypeSupported(t));
    const rec = new MediaRecorder(stream, mimeType ? { mimeType, audioBitsPerSecond: 32_000 } : undefined);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    rec.onstop = () => {
      const blob = new Blob(chunks, { type: rec.mimeType || mimeType || 'audio/webm' });
      release();
      setState({ kind: 'idle' });
      if (!cancelledRef.current && blob.size > 0) onCompleteRef.current(blob);
    };
    rec.onerror = () => {
      release();
      setState({ kind: 'error', reason: 'failed' });
    };
    recRef.current = rec;
    rec.start(250);
    setState({ kind: 'recording', startedAt: Date.now() });
    timerRef.current = setTimeout(() => {
      if (rec.state === 'recording') rec.stop();
    }, maxSeconds * 1000);
  }, [maxSeconds, release]);

  const stop = useCallback(() => {
    if (recRef.current?.state === 'recording') recRef.current.stop();
  }, []);

  const cancel = useCallback(() => {
    cancelledRef.current = true;
    stop();
  }, [stop]);

  const dismissError = useCallback(() => {
    setState({ kind: 'idle' });
  }, []);

  return { state, start, stop, cancel, dismissError };
}
