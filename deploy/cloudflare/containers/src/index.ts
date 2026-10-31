import { Container, getContainer } from "@cloudflare/containers";

interface Env {
  RESEARCHGAP: DurableObjectNamespace<ResearchGapApp>;
  GEMINI_API_KEY: string; // set with `wrangler secret put GEMINI_API_KEY`
  GEMMA_MODEL?: string;
}

export class ResearchGapApp extends Container<Env> {
  defaultPort = 8501; // Streamlit
  sleepAfter = "20m";

  constructor(ctx: DurableObjectState<{}>, env: Env) {
    super(ctx, env);
    this.envVars = {
      GEMINI_API_KEY: env.GEMINI_API_KEY,
      GEMMA_MODEL: env.GEMMA_MODEL ?? "gemma-4-31b-it",
    };
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // One shared instance: Streamlit keeps a separate session per browser tab,
    // and its WebSocket upgrade is forwarded as-is.
    return getContainer(env.RESEARCHGAP, "main").fetch(request);
  },
};
