import { useQuery } from '@tanstack/react-query';

import { useApi } from './api-context';

export const queryKeys = {
  config: ['config'] as const,
  health: ['health'] as const,
};

/** Backend capabilities; effectively static for a deployment. */
export function useConfig() {
  const api = useApi();
  return useQuery({
    queryKey: queryKeys.config,
    queryFn: ({ signal }) => api.getConfig(signal),
    staleTime: Infinity,
    retry: 2,
  });
}

/** Polled so the header reflects cold starts / a degraded index. */
export function useHealth() {
  const api = useApi();
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: ({ signal }) => api.getHealth(signal),
    refetchInterval: 60_000,
    retry: 1,
  });
}
