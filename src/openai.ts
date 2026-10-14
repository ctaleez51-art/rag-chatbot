// 모두콘 안내 챗봇 — OpenAI API 클라이언트 (사용자 API 키, 브라우저 직접 호출)
import { judgeAll, type JudgeResult } from "./judge";
import type { ChatMsg } from "./ollama";

const BASE = "https://api.openai.com/v1";

/** 키로 쓸 수 있는 채팅 모델 목록. 모델 이름을 짐작해 적지 않고 API에서 받아 온다. */
export async function listOpenAIModels(apiKey: string): Promise<string[]> {
  const res = await fetch(`${BASE}/models`, { headers: { Authorization: `Bearer ${apiKey}` } });
  if (!res.ok) throw new Error(`openai models ${res.status}`);
  const j = await res.json();
  const skip = /embedding|tts|whisper|dall-e|image|audio|realtime|moderation|transcribe|search/;
  return (j.data ?? [])
    .map((m: { id: string }) => m.id)
    .filter((id: string) => !skip.test(id))
    .sort();
}

/** SSE 스트리밍 Chat Completions. */
export async function openaiStream(
  messages: ChatMsg[],
  apiKey: string,
  model: string,
  onToken: (t: string) => void,
  signal?: AbortSignal,
): Promise<string> {
  const res = await fetch(`${BASE}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model, messages, stream: true }),
    signal,
  });
  if (!res.ok) throw Object.assign(new Error(`openai ${res.status}`), { status: res.status });
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let full = "";
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (!payload || payload === "[DONE]") continue;
      try {
        const piece: string = JSON.parse(payload).choices?.[0]?.delta?.content ?? "";
        if (piece) {
          full += piece;
          onToken(piece);
        }
      } catch { /* 불완전 라인 */ }
    }
  }
  return full;
}

/** LLM-as-a-Judge(OpenAI): 기준·병렬·집계는 judge.ts 공통 로직 사용. */
export async function judgeWithOpenAI(
  question: string,
  sources: string,
  answer: string,
  apiKey: string,
  model: string,
): Promise<JudgeResult> {
  const call = async (prompt: string): Promise<string> => {
    const res = await fetch(`${BASE}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }] }),
    });
    if (!res.ok) throw new Error(`openai judge ${res.status}`);
    const j = await res.json();
    return j.choices?.[0]?.message?.content ?? "";
  };
  return judgeAll(question, sources, answer, call);
}
