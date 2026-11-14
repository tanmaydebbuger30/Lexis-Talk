import { Fragment, type ReactNode } from 'react';

const CITE = /\[(S\d+)\]/g;

/**
 * Renders model output as plain text (never HTML), turning [S1] markers into
 * buttons that open the matching source. No markdown/HTML rendering means no
 * XSS surface from model output or from retrieved document text.
 */
export function AnswerText({
  text,
  knownIds,
  onCite,
}: {
  text: string;
  knownIds: ReadonlySet<string>;
  onCite: (sourceId: string) => void;
}) {
  const paragraphs = text.split(/\n{2,}/);
  return (
    <>
      {paragraphs.map((p, pi) => {
        const parts: ReactNode[] = [];
        let last = 0;
        for (const m of p.matchAll(CITE)) {
          const id = m[1] ?? '';
          parts.push(p.slice(last, m.index));
          parts.push(
            knownIds.has(id) ? (
              <button
                key={`${pi}-${m.index}`}
                type="button"
                onClick={() => {
                  onCite(id);
                }}
                className="mx-0.5 inline-flex -translate-y-0.5 items-center rounded bg-amber-100 px-1 font-mono text-[11px] font-semibold text-amber-900 hover:bg-amber-200 focus-visible:outline-2 focus-visible:outline-amber-600"
                aria-label={`Show source ${id}`}
              >
                {id}
              </button>
            ) : (
              m[0]
            ),
          );
          last = m.index + m[0].length;
        }
        parts.push(p.slice(last));
        return (
          <p key={pi} className="whitespace-pre-wrap [&:not(:first-child)]:mt-3">
            {parts.map((x, i) => (
              <Fragment key={i}>{x}</Fragment>
            ))}
          </p>
        );
      })}
    </>
  );
}
