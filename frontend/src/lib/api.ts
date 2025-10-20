import { OpenAPI } from '../client';

const DEFAULT_API_BASE = 'http://localhost:8000';

const sanitizeBaseUrl = (value: string): string => value.replace(/\/+$/, '');

const apiBaseEnv = import.meta.env.VITE_API_BASE_URL;
const apiBase = sanitizeBaseUrl(apiBaseEnv && apiBaseEnv.length > 0 ? apiBaseEnv : DEFAULT_API_BASE);

OpenAPI.BASE = apiBase;
OpenAPI.WITH_CREDENTIALS = false;

export const API_BASE_URL = apiBase;

const resolveWsBase = (): string => {
  const wsEnv = import.meta.env.VITE_WS_BASE_URL;
  if (wsEnv && wsEnv.length > 0) {
    return sanitizeBaseUrl(wsEnv);
  }

  try {
    const url = new URL(apiBase);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    return sanitizeBaseUrl(url.toString());
  } catch {
    return sanitizeBaseUrl(apiBase.replace(/^http/, 'ws'));
  }
};

export const buildApiUrl = (path: string): string => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
};

export const buildStorageUrl = (relativePath: string): string => {
  const trimmed = relativePath.replace(/^\/+/, '');
  return buildApiUrl(`/storage/${trimmed}`);
};

export const WS_BASE_URL = resolveWsBase();
