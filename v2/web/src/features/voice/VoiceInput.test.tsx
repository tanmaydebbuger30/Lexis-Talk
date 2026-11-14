import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

import { API, errorBody } from '../../test/fixtures';
import { renderApp } from '../../test/render';
import { server } from '../../test/server';
import { pickMimeType } from './useRecorder';

class FakeMediaRecorder {
  static isTypeSupported = (t: string) => t.startsWith('audio/webm');
  static instances: FakeMediaRecorder[] = [];
  state: 'inactive' | 'recording' = 'inactive';
  mimeType: string;
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(_s: MediaStream, opts?: { mimeType?: string }) {
    this.mimeType = opts?.mimeType ?? 'audio/webm';
    FakeMediaRecorder.instances.push(this);
  }
  start() {
    this.state = 'recording';
  }
  stop() {
    this.state = 'inactive';
    this.ondataavailable?.({ data: new Blob([new Uint8Array(2048)], { type: this.mimeType }) });
    this.onstop?.();
  }
}

const trackStop = vi.fn();

function installMedia(getUserMedia: () => Promise<MediaStream>) {
  vi.stubGlobal('MediaRecorder', FakeMediaRecorder);
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia } });
}

const okStream = () => Promise.resolve({ getTracks: () => [{ stop: trackStop }] } as unknown as MediaStream);

afterEach(() => {
  vi.unstubAllGlobals();
  Reflect.deleteProperty(navigator, 'mediaDevices');
  FakeMediaRecorder.instances = [];
  trackStop.mockReset();
});

const composer = () => screen.getByRole('textbox', { name: /ask a question/i });

describe('voice input', () => {
  it('records, transcribes, and puts the transcript in the composer for review', async () => {
    installMedia(okStream);
    let chatCalled = false;
    let uploadedType = '';
    server.use(
      http.post(`${API}/api/transcribe`, async ({ request }) => {
        const audio = (await request.formData()).get('audio');
        uploadedType = typeof audio === 'object' && audio ? audio.type : '';
        return HttpResponse.json({ text: 'what does title 3 say', input_lang: 'auto', duration_seconds: 2 });
      }),
      http.post(`${API}/api/chat/stream`, () => {
        chatCalled = true;
        return HttpResponse.json({}, { status: 500 });
      }),
    );
    const { user } = renderApp();
    const mic = await screen.findByRole('button', { name: 'Record a question' });
    await waitFor(() => {
      expect(mic).toBeEnabled();
    });

    await user.click(mic);
    const stop = await screen.findByRole('button', { name: 'Stop recording' });
    expect(stop).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('Recording…')).toBeInTheDocument();

    await user.click(stop);
    await waitFor(() => {
      expect(composer()).toHaveValue('what does title 3 say');
    });
    expect(screen.getByText(/Transcribed from your recording/)).toBeInTheDocument();
    expect(uploadedType).toBe('audio/webm;codecs=opus');
    expect(trackStop).toHaveBeenCalled(); // microphone released
    expect(chatCalled).toBe(false); // nothing is sent until the user presses Send
  });

  it('explains when microphone permission is denied', async () => {
    installMedia(() => Promise.reject(new DOMException('denied', 'NotAllowedError')));
    const { user } = renderApp();
    const mic = await screen.findByRole('button', { name: 'Record a question' });
    await waitFor(() => {
      expect(mic).toBeEnabled();
    });
    await user.click(mic);
    expect(await screen.findByText(/Microphone access was blocked/)).toBeInTheDocument();
  });

  it('shows transcription errors from the server', async () => {
    installMedia(okStream);
    server.use(
      http.post(`${API}/api/transcribe`, () =>
        HttpResponse.json(errorBody('bad_audio', 'Recording is too short.'), { status: 422 }),
      ),
    );
    const { user } = renderApp();
    const mic = await screen.findByRole('button', { name: 'Record a question' });
    await waitFor(() => {
      expect(mic).toBeEnabled();
    });
    await user.click(mic);
    await user.click(await screen.findByRole('button', { name: 'Stop recording' }));
    expect(await screen.findByText('Recording is too short.')).toBeInTheDocument();
    expect(composer()).toHaveValue('');
  });

  it('disables the mic when the browser cannot record', async () => {
    renderApp();
    const mic = await screen.findByRole('button', { name: 'Record a question' });
    expect(mic).toBeDisabled();
    expect(mic).toHaveAttribute('title', 'Voice input is not supported in this browser.');
  });

  it('prefers Opus in WebM, then falls back', () => {
    expect(pickMimeType(() => true)).toBe('audio/webm;codecs=opus');
    expect(pickMimeType((t) => t === 'audio/mp4')).toBe('audio/mp4');
    expect(pickMimeType(() => false)).toBeUndefined();
  });
});
