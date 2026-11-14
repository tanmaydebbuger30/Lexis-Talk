import { render } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { createApiClient } from '../api/client';
import { App } from '../app/App';
import { API } from './fixtures';

export function renderApp(opts: { timeoutMs?: number } = {}) {
  const user = userEvent.setup();
  const client = createApiClient(API, opts);
  const utils = render(<App client={client} />);
  return { user, ...utils };
}
