import { useState } from 'react';

import { type InputLang, InputLangSchema, type OutputLang, OutputLangSchema } from '../../api/schemas';

const KEY = 'lexitalk.lang.v1';

export type LanguagePrefs = { input: InputLang; output: OutputLang };

/** Only the language preference is kept in localStorage; never chat content. */
function load(): LanguagePrefs {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? '{}') as { input?: unknown; output?: unknown };
    return {
      input: InputLangSchema.catch('auto').parse(raw.input),
      output: OutputLangSchema.catch('en').parse(raw.output),
    };
  } catch {
    return { input: 'auto', output: 'en' };
  }
}

export function useLanguagePrefs() {
  const [prefs, setPrefs] = useState<LanguagePrefs>(load);
  const update = (p: Partial<LanguagePrefs>) => {
    setPrefs((prev) => {
      const next = { ...prev, ...p };
      try {
        localStorage.setItem(KEY, JSON.stringify(next));
      } catch {
        /* ignore */
      }
      return next;
    });
  };
  return [prefs, update] as const;
}
