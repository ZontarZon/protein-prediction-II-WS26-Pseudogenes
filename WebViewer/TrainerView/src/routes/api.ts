// Relative URLs use the host the browser actually opened. This supports both
// Flask hosting and LAN access to the Vite dev server (which proxies /api).
const API = import.meta.env.VITE_API_URL ?? "/api";
export function apiUrl(path: string) {
  return `${API}${path.startsWith("/api") ? path.slice(4) : path}`;
}
export async function get<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path));
  const value = await response.json();
  if (!response.ok)
    throw new Error(
      [
        value.error ?? `Request failed (${response.status})`,
        ...(value.errors ?? []),
      ].join(": "),
    );
  return value as T;
}
export async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok)
    throw new Error(
      [
        value.error ?? `Request failed (${response.status})`,
        ...(value.errors ?? []),
      ].join(": "),
    );
  return value as T;
}
export async function remove<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path), { method: "DELETE" });
  const value = await response.json();
  if (!response.ok)
    throw new Error(
      [
        value.error ?? `Request failed (${response.status})`,
        ...(value.errors ?? []),
      ].join(": "),
    );
  return value as T;
}
export function useResource<T>(path: string, key: string, pollMs = 0) {
  const identity = `${path}\u0000${key}`;
  const [state, setState] = React.useState<{
    identity: string;
    data?: T;
    error?: string;
  }>({ identity: "", data: undefined });
  React.useEffect(() => {
    let alive = true;
    let busy = false;
    const load = async () => {
      if (busy) return;
      busy = true;
      try {
        const data = await get<T>(path);
        if (alive) setState({ identity, data });
      } catch (error) {
        if (alive) setState({ identity, error: (error as Error).message });
      } finally { busy = false; }
    };
    void load();
    const timer = pollMs > 0 ? window.setInterval(() => void load(), pollMs) : undefined;
    return () => {
      alive = false;
      if (timer !== undefined) window.clearInterval(timer);
    };
  }, [identity, path, pollMs]);
  const current = state.identity === identity;
  return {
    data: current ? state.data : undefined,
    error: current ? state.error : undefined,
    loading: !current,
  };
}
import * as React from "react";
