// GET /api/prompts: the prompt registry (src/prompts) and the commit the server runs.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { DatabaseSync } from "node:sqlite";
import type { PromptsReport } from "../../../web/src/api/types.ts";
import { promptsReport, type ProviderMode } from "../../prompts/index.ts";
import { classifierProvider, modelProvider } from "../../agents/compatible.ts";
import { inferenceMode } from "../../remote/mode.ts";

// Read when the server loads its code, so a later checkout without a restart
// does not change what the page claims.
const COMMIT = ((): string | null => {
  try {
    const cwd = fileURLToPath(new URL("../../..", import.meta.url));
    // A deployed release has no .git; the deploy hook writes the commit here.
    try {
      const stamped = readFileSync(`${cwd}/.source-commit`, "utf8").trim();
      if (/^[0-9a-f]{40}$/.test(stamped)) return stamped;
    } catch {}
    return execFileSync("git", ["rev-parse", "HEAD"], { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 5000 }).trim() || null;
  } catch {
    return null;
  }
})();

export function prompts(db: DatabaseSync): PromptsReport {
  // Which provider answers now, so the page describes the calls this host makes, not the default ones. In the user-local
  // mode neither provider runs here: helpers run the calls, with the command shown.
  if (inferenceMode(db) === "user-local") return promptsReport(COMMIT);
  const mode: ProviderMode = {
    model: modelProvider().provider === "compatible" ? "compatible" : "claude",
    classifier: classifierProvider().provider === "compatible" ? "compatible" : "jev",
  };
  return promptsReport(COMMIT, mode);
}
