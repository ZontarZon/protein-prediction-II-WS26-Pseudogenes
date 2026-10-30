import type { ReactNode } from "react";
export function Page({
  eyebrow,
  title,
  description,
  action,
  children,
}: {
  eyebrow: string;
  title: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          {description && <p>{description}</p>}
        </div>
        {action && <div>{action}</div>}
      </div>
      {children}
    </>
  );
}
export function Panel({
  title,
  meta,
  children,
}: {
  title: string;
  meta?: string;
  children: ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>{title}</h2>
        {meta && <span className="mono muted">{meta}</span>}
      </div>
      {children}
    </section>
  );
}
export function State({
  loading,
  error,
  empty,
}: {
  loading?: boolean;
  error?: string;
  empty?: string;
}) {
  if (loading) return <div className="state">Scanning workspace…</div>;
  if (error)
    return (
      <div className="state error">
        {error}
        <small>Check that the backend is running and reachable at /api/health.</small>
      </div>
    );
  if (empty) return <div className="state">{empty}</div>;
  return null;
}
