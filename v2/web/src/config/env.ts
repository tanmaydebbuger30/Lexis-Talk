import { z } from 'zod';

/**
 * Build-time public configuration. VITE_* values are inlined into the bundle,
 * so nothing here may be secret. Validated once at startup so a bad deploy
 * fails loudly instead of sending requests to `undefined/api/chat`.
 */
const EnvSchema = z.object({
  VITE_API_BASE_URL: z
    .url({ protocol: /^https?$/ })
    .transform((u) => u.replace(/\/+$/, ''))
    .refine(
      (u) =>
        !import.meta.env.PROD ||
        u.startsWith('https://') ||
        /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(u),
      {
        message: 'VITE_API_BASE_URL must use https in production builds (http is allowed only for localhost)',
      },
    ),
});

export type Env = { apiBaseUrl: string };

export function readEnv(source: Record<string, unknown> = import.meta.env): Env {
  const parsed = EnvSchema.safeParse(source);
  if (!parsed.success) {
    throw new Error(
      `Invalid environment configuration: ${parsed.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ')}`,
    );
  }
  return { apiBaseUrl: parsed.data.VITE_API_BASE_URL };
}
