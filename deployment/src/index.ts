import { Container, getContainer } from "@cloudflare/containers";

export interface Env {
  FRAUD_DETECTOR: DurableObjectNamespace<FraudDetectorContainer>;
}

export class FraudDetectorContainer extends Container<Env> {
  defaultPort = 8080;
  sleepAfter = "5m";

  override onError(error: unknown) {
    console.log("Container error:", error);
  }
}

const CORS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function withCors(res: Response): Response {
  const headers = new Headers(res.headers);
  for (const [k, v] of Object.entries(CORS)) headers.set(k, v);
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS });
    }
    const container = getContainer(env.FRAUD_DETECTOR);
    const resp = await container.fetch(req);
    return withCors(resp);
  },
} satisfies ExportedHandler<Env>;
