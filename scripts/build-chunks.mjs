// 자료 md 파일 2개를 조각(청크)으로 잘라 data/chunks.json에 저장한다.
// 사용법 (rag-chatbot 폴더에서): node scripts/build-chunks.mjs
// 자료 파일을 고쳤으면 이것을 다시 실행하고, 이어서 embed-docs.mjs도 다시 실행한다.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SOURCE = join(ROOT, "source");
const QNA_FILE = join(SOURCE, "국세상담-인적공제QnA.md");
const KW_FILE = join(SOURCE, "키워드연말정산-인적공제.md");
const QNA_URL = "https://call.nts.go.kr/call/qna/selectQnaInfo.do?mi=1318";
const KW_SOURCE = "2025년 귀속 키워드연말정산게시용.hwp";

const read = (f) => readFileSync(f, "utf8").replace(/\r\n/g, "\n");
const pad = (n) => String(n).padStart(3, "0");

// 국세상담 Q&A: #### 질문 하나 = 조각 하나
function qnaChunks() {
  const out = [];
  let group = "", sub = "", title = null, body = [];
  const flush = () => {
    if (title === null) return;
    const text = body.join("\n").trim();
    out.push({
      id: `QA-${pad(out.length + 1)}`,
      text,
      url: QNA_URL,
      section: ["국세상담 Q&A", group, sub, title].filter(Boolean).join(" · "),
    });
    title = null; body = [];
  };
  for (const line of read(QNA_FILE).split("\n")) {
    if (line.startsWith("#### ")) { flush(); title = line.slice(5).trim(); continue; }
    if (line.startsWith("### ")) { flush(); sub = line.slice(4).trim(); continue; }
    if (line.startsWith("## ")) { flush(); group = line.slice(3).trim(); sub = ""; continue; }
    if (line.startsWith("# ")) continue;
    if (title !== null) body.push(line);
  }
  flush();
  return out;
}

// 키워드 연말정산: 항목(##) 안에서 ○로 시작하는 문단 하나 = 조각 하나
// (○ 앞에 글이 있으면 그것도 조각 하나, ○ 뒤의 표·- 줄은 그 ○ 조각에 붙는다)
function kwChunks() {
  const out = [];
  let item = null, body = [];
  const flush = () => {
    const text = body.join("\n").trim();
    if (item && text) {
      out.push({
        id: `KW-${pad(out.length + 1)}`,
        text,
        url: `${KW_SOURCE} · 항목 ${item.split(" ")[0]}`,
        section: `키워드 연말정산 · ${item}`,
      });
    }
    body = [];
  };
  for (const line of read(KW_FILE).split("\n")) {
    if (line.startsWith("## ")) { flush(); item = line.slice(3).trim(); continue; }
    if (!item) continue; // 파일 맨 위 설명
    if (line.startsWith("○")) flush();
    body.push(line);
  }
  flush();
  return out;
}

const chunks = [...qnaChunks(), ...kwChunks()];
const short = chunks.filter((c) => c.text.length < 120);
writeFileSync(join(ROOT, "data", "chunks.json"), JSON.stringify(chunks, null, 2) + "\n");
console.log(`국세상담 Q&A ${chunks.filter((c) => c.id.startsWith("QA")).length}개, 키워드 연말정산 ${chunks.filter((c) => c.id.startsWith("KW")).length}개 → data/chunks.json`);
console.log(`120자보다 짧은 조각: ${short.length}개`);
