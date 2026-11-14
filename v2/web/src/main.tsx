import './index.css';

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { createApiClient } from './api/client';
import { App } from './app/App';
import { readEnv } from './config/env';

const root = document.getElementById('root');
if (!root) throw new Error('#root not found');

const env = readEnv();
createRoot(root).render(
  <StrictMode>
    <App client={createApiClient(env.apiBaseUrl)} />
  </StrictMode>,
);
