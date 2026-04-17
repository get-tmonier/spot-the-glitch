import { ImageResponse } from "@vercel/og";
import type { NextRequest } from "next/server";

export const runtime = "edge";

export function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const score = searchParams.get("score") ?? "0";

  return new ImageResponse(
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        width: "100%",
        height: "100%",
        backgroundColor: "#0a0a0a",
        color: "#ededed",
        fontFamily: "system-ui",
      }}
    >
      <div style={{ fontSize: 64, fontWeight: 700 }}>Spot the Glitch</div>
      <div style={{ fontSize: 128, fontWeight: 700, color: "#3b82f6", marginTop: 24 }}>
        {score}/10
      </div>
      <div style={{ fontSize: 32, opacity: 0.7, marginTop: 16 }}>
        AI scored 10/10. Can you do better?
      </div>
    </div>,
    { width: 1200, height: 630 },
  );
}
