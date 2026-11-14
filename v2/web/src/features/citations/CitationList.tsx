import { useId, useState } from 'react';

import type { Source } from '../../api/schemas';
import { citationAnchorId } from './anchors';

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  return (
    <span className="inline-flex items-center gap-1.5" title="Heuristic retrieval score, not a probability">
      <span aria-hidden="true" className="h-1.5 w-12 overflow-hidden rounded-full bg-stone-200">
        <span className="block h-full rounded-full bg-amber-600" style={{ width: `${pct}%` }} />
      </span>
      <span className="tabular-nums">
        <span className="sr-only">Relevance score </span>
        {score.toFixed(2)}
      </span>
    </span>
  );
}

function CitationCard({
  source,
  messageId,
  open,
  onToggle,
}: {
  source: Source;
  messageId: string;
  open: boolean;
  onToggle: () => void;
}) {
  const panelId = useId();
  return (
    <li
      id={citationAnchorId(messageId, source.id)}
      className={`scroll-mt-24 rounded-lg border bg-white transition-colors ${
        source.used_in_prompt ? 'border-stone-300' : 'border-dashed border-stone-300 opacity-80'
      }`}
    >
      <h4>
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          aria-controls={panelId}
          className="flex w-full items-start gap-3 rounded-lg px-3 py-2.5 text-left hover:bg-stone-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-600"
        >
          <span className="mt-0.5 rounded bg-stone-900 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-amber-300">
            {source.id}
          </span>
          <span className="min-w-0 flex-1">
            <span className="block font-medium text-stone-900">{source.title}</span>
            <span className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-stone-500">
              {source.page !== null && <span>Page {source.page}</span>}
              <ScoreBar score={source.score} />
              <span>{source.used_in_prompt ? 'Used in answer' : 'Retrieved, not used'}</span>
            </span>
          </span>
          <svg
            aria-hidden="true"
            viewBox="0 0 20 20"
            className={`mt-1 h-4 w-4 shrink-0 text-stone-400 transition-transform ${open ? 'rotate-180' : ''}`}
          >
            <path
              d="M5 8l5 5 5-5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </h4>
      <div id={panelId} hidden={!open} className="border-t border-stone-200 px-3 py-3">
        <blockquote className="font-serif text-[15px] leading-relaxed whitespace-pre-wrap text-stone-700">
          {source.snippet}
        </blockquote>
      </div>
    </li>
  );
}

export function CitationList({
  sources,
  messageId,
  openIds,
  onToggle,
}: {
  sources: Source[];
  messageId: string;
  openIds: ReadonlySet<string>;
  onToggle: (sourceId: string) => void;
}) {
  const [showUnused, setShowUnused] = useState(false);
  const used = sources.filter((s) => s.used_in_prompt);
  const unused = sources.filter((s) => !s.used_in_prompt);
  const visible = showUnused ? sources : used;
  if (sources.length === 0) return null;

  return (
    <section aria-label="Sources" className="mt-4">
      <h3 className="mb-2 text-xs font-semibold tracking-wider text-stone-500 uppercase">
        Sources ({used.length} used)
      </h3>
      <ol className="space-y-2">
        {visible.map((s) => (
          <CitationCard
            key={s.id}
            source={s}
            messageId={messageId}
            open={openIds.has(s.id)}
            onToggle={() => {
              onToggle(s.id);
            }}
          />
        ))}
      </ol>
      {unused.length > 0 && (
        <button
          type="button"
          onClick={() => {
            setShowUnused((v) => !v);
          }}
          className="mt-2 text-xs font-medium text-stone-600 underline decoration-stone-300 underline-offset-2 hover:text-stone-900"
        >
          {showUnused ? 'Hide' : 'Show'} {unused.length} retrieved passage{unused.length === 1 ? '' : 's'} not
          used
        </button>
      )}
    </section>
  );
}
