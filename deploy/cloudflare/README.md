# Hosting ResearchGap on Cloudflare

Streamlit is a long-running Python server that talks to the browser over a WebSocket, so it cannot run on
Cloudflare Pages or plain Workers. There are two ways to put it behind Cloudflare.

| | Option A: Cloudflare Tunnel (recommended) | Option B: Cloudflare Containers |
|---|---|---|
| Where the app runs | Your laptop, a VM, or any Docker host | Cloudflare's network |
| Cost | Free | Workers Paid plan |
| Setup time | 2 minutes (quick tunnel) | ~15 minutes |
| Needs Docker locally | No (yes for compose) | Yes, to build the image |

## Option A: Cloudflare Tunnel

### A1. Quick tunnel (no account, good for a hackathon demo)

1. Start the app locally (see the main README): `streamlit run app.py`
2. Install cloudflared: `winget install --id Cloudflare.cloudflared` (Windows) or `brew install cloudflared` (macOS).
3. Run:
   ```bash
   cloudflared tunnel --url http://localhost:8501
   ```
4. Share the `https://<random>.trycloudflare.com` URL it prints. It stays up while the command runs.

### A2. Named tunnel on your own domain, with Docker Compose

1. In the Cloudflare dashboard go to **Zero Trust → Networks → Tunnels → Create a tunnel**, choose
   **Cloudflared**, and copy the tunnel token.
2. Add a **Public hostname** (e.g. `researchgap.yourdomain.com`) with service `http://app:8501`.
3. Put the token in `.env` as `CLOUDFLARE_TUNNEL_TOKEN=...` next to `GEMINI_API_KEY`.
4. From the repo root:
   ```bash
   docker compose up -d --build
   ```
   The `tunnel` service connects out to Cloudflare, so no ports need to be opened.

Optional: put **Cloudflare Access** in front of the hostname so only invited emails can use your Gemini quota.

## Option B: Cloudflare Containers

The Worker in `containers/` forwards every request (including Streamlit's WebSocket) to a container built
from the repo's `Dockerfile`.

```bash
cd deploy/cloudflare/containers
npm install
npx wrangler login
npx wrangler secret put GEMINI_API_KEY
npx wrangler deploy
```

Docker Desktop must be running so wrangler can build the image. The first request after the container
sleeps (`sleepAfter = "20m"`) takes a few seconds while it cold-starts. Raise `instance_type` in
`wrangler.jsonc` if large PDFs run out of memory.
