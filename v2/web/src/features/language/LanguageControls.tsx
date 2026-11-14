import { type InputLang, InputLangSchema, type OutputLang, OutputLangSchema } from '../../api/schemas';
import type { LanguagePrefs } from './prefs';

const INPUT_LABEL: Record<InputLang, string> = {
  auto: 'Auto',
  en: 'English',
  hi: 'हिन्दी (Hindi)',
};
const OUTPUT_LABEL: Record<OutputLang, string> = { en: 'English', hi: 'हिन्दी (Hindi)' };
const selectCls =
  'min-w-0 flex-1 rounded-md border border-stone-300 bg-white py-1.5 pr-8 pl-2.5 text-sm text-stone-900 focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-amber-600 disabled:opacity-50';

export function LanguageControls({
  prefs,
  onChange,
  inputOptions,
  outputOptions,
  disabled,
}: {
  prefs: LanguagePrefs;
  onChange: (p: Partial<LanguagePrefs>) => void;
  inputOptions: readonly InputLang[];
  outputOptions: readonly OutputLang[];
  disabled?: boolean;
}) {
  return (
    <fieldset className="grid grid-cols-2 gap-3 sm:flex sm:items-center sm:gap-4" disabled={disabled}>
      <legend className="sr-only">Languages</legend>
      <label className="flex min-w-0 items-center gap-2 text-xs font-medium text-stone-600">
        Ask in
        <select
          className={selectCls}
          value={prefs.input}
          onChange={(e) => {
            onChange({ input: InputLangSchema.parse(e.target.value) });
          }}
        >
          {inputOptions.map((l) => (
            <option key={l} value={l}>
              {INPUT_LABEL[l]}
            </option>
          ))}
        </select>
      </label>
      <label className="flex min-w-0 items-center gap-2 text-xs font-medium text-stone-600">
        Reply in
        <select
          className={selectCls}
          value={prefs.output}
          onChange={(e) => {
            onChange({ output: OutputLangSchema.parse(e.target.value) });
          }}
        >
          {outputOptions.map((l) => (
            <option key={l} value={l}>
              {OUTPUT_LABEL[l]}
            </option>
          ))}
        </select>
      </label>
    </fieldset>
  );
}
