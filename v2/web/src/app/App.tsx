import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { type ReactNode, useState } from 'react';

import type { ApiClient } from '../api/client';
import { ChatPage } from '../features/chat/ChatPage';
import { ApiContext } from './api-context';

export function Providers({ client, children }: { client: ApiClient; children: ReactNode }) {
  const [qc] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { refetchOnWindowFocus: false }, mutations: { retry: 0 } },
      }),
  );
  return (
    <QueryClientProvider client={qc}>
      <ApiContext.Provider value={client}>{children}</ApiContext.Provider>
    </QueryClientProvider>
  );
}

export function App({ client }: { client: ApiClient }) {
  return (
    <Providers client={client}>
      <ChatPage />
    </Providers>
  );
}
