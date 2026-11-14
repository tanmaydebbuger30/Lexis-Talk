import { setupServer } from 'msw/node';

import { defaultHandlers } from './fixtures';

export const server = setupServer(...defaultHandlers);
