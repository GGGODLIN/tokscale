import type { Metadata } from "next";
import { TOKSCALE_X_HANDLE } from "@/lib/community";
import { getPublicOrigin } from "./urls";

export function getRootMetadata(origin = getPublicOrigin()): Metadata {
  // Bump the query when og-image.png changes so social crawlers refetch it.
  const ogImage = `${origin}/og-image.png?v=3`;

  return {
    metadataBase: new URL(origin),
    title: "Tokscale - AI Token Usage Tracker & Leaderboard",
    description: "Track, visualize, and compete on AI coding assistant token usage across Claude Code, Cursor, OpenCode, Codex, Gemini, Kimi, and Qwen. The Kardashev Scale for AI Devs.",
    icons: {
      icon: [
        { url: "/favicon-16x16.png", sizes: "16x16", type: "image/png" },
        { url: "/favicon-32x32.png", sizes: "32x32", type: "image/png" },
      ],
      apple: "/apple-icon.png",
    },
    manifest: "/site.webmanifest",
    openGraph: {
      title: "Tokscale - AI Token Usage Tracker & Leaderboard",
      description: "Track, visualize, and compete on AI coding assistant token usage across Claude Code, Cursor, OpenCode, Codex, Gemini, Kimi, and Qwen. The Kardashev Scale for AI Devs.",
      type: "website",
      url: origin,
      siteName: "Tokscale",
      images: [{ url: ogImage, width: 2400, height: 1260, alt: "Tokscale - AI Token Usage Tracker" }],
    },
    twitter: {
      card: "summary_large_image",
      site: TOKSCALE_X_HANDLE,
      title: "Tokscale - AI Token Usage Tracker & Leaderboard",
      description: "Track, visualize, and compete on AI coding assistant token usage across Claude Code, Cursor, OpenCode, Codex, Gemini, Kimi, and Qwen.",
      images: [ogImage],
    },
  };
}
