import { readEnv } from './env';

describe('readEnv', () => {
  it('normalises a trailing slash', () => {
    expect(readEnv({ VITE_API_BASE_URL: 'https://api.example.com/' })).toEqual({
      apiBaseUrl: 'https://api.example.com',
    });
  });

  it.each([undefined, '', 'not a url', 'ftp://x.example', 'javascript:alert(1)'])('rejects %s', (v) => {
    expect(() => readEnv({ VITE_API_BASE_URL: v })).toThrow(/Invalid environment configuration/);
  });
});
