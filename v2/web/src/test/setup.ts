import '@testing-library/jest-dom/vitest';

import { cleanup } from '@testing-library/react';

import { server } from './server';

beforeAll(() => {
  server.listen({ onUnhandledFrame: 'error' });
});
afterEach(() => {
  cleanup();
  server.resetHandlers();
  sessionStorage.clear();
  localStorage.clear();
});
afterAll(() => {
  server.close();
});

// jsdom does not implement scrollIntoView
Element.prototype.scrollIntoView = function scrollIntoView() {};
