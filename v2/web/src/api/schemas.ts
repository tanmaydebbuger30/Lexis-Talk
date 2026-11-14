import { z } from 'zod';

/**
 * Runtime schemas for every backend response. They mirror api/lexitalk/schemas.py
 * exactly; responses are parsed (not cast), so a backend contract change surfaces
 * as a typed `contract_mismatch` error instead of `undefined` deep in the UI.
 */

export const InputLangSchema = z.enum(['auto', 'en', 'hi']);
export const OutputLangSchema = z.enum(['en', 'hi']);
export const DetectedLangSchema = z.enum(['en', 'hi']);
export type InputLang = z.infer<typeof InputLangSchema>;
export type OutputLang = z.infer<typeof OutputLangSchema>;

export const SourceSchema = z.object({
  id: z.string(),
  document_id: z.string(),
  title: z.string(),
  page: z.number().int().nullable(),
  snippet: z.string(),
  score: z.number().min(0).max(1),
  rank: z.number().int().positive(),
  used_in_prompt: z.boolean(),
});
export type Source = z.infer<typeof SourceSchema>;

export const RedactionSchema = z.object({ type: z.string(), count: z.number().int().positive() });
export type Redaction = z.infer<typeof RedactionSchema>;

export const ChatResponseSchema = z.object({
  answer: z.string(),
  detected_lang: DetectedLangSchema,
  query_used_for_retrieval: z.string(),
  grounded: z.boolean(),
  sources: z.array(SourceSchema),
  redactions: z.array(RedactionSchema),
  request_id: z.string(),
});
export type ChatResponse = z.infer<typeof ChatResponseSchema>;

export const StreamMetaSchema = z.object({
  request_id: z.string(),
  detected_lang: DetectedLangSchema,
  query_used_for_retrieval: z.string(),
  grounded: z.boolean(),
  sources: z.array(SourceSchema),
});
export type StreamMeta = z.infer<typeof StreamMetaSchema>;
export const StreamTokenSchema = z.object({ delta: z.string() });
export const StreamDoneSchema = z.object({ redactions: z.array(RedactionSchema) });

export const ErrorResponseSchema = z.object({
  error: z.object({ code: z.string(), message: z.string(), request_id: z.string() }),
});
export const StreamErrorSchema = ErrorResponseSchema.shape.error;

export const TranscribeResponseSchema = z.object({
  text: z.string(),
  input_lang: InputLangSchema,
  duration_seconds: z.number().nonnegative(),
});
export type TranscribeResponse = z.infer<typeof TranscribeResponseSchema>;

export const ConfigResponseSchema = z.object({
  input_languages: z.array(InputLangSchema),
  output_languages: z.array(OutputLangSchema),
  tts_enabled: z.boolean(),
  max_message_chars: z.number().int().positive(),
  max_audio_seconds: z.number().int().positive(),
  max_audio_bytes: z.number().int().positive(),
});
export type ConfigResponse = z.infer<typeof ConfigResponseSchema>;

export const HealthResponseSchema = z.object({
  status: z.enum(['ok', 'degraded']),
  index_ready: z.boolean(),
  chunks_indexed: z.number().int().nonnegative(),
  version: z.string(),
});
export type HealthResponse = z.infer<typeof HealthResponseSchema>;

export type HistoryTurn = { role: 'user' | 'assistant'; content: string };
export type ChatRequest = {
  message: string;
  input_lang: InputLang;
  output_lang: OutputLang;
  history: HistoryTurn[];
};
