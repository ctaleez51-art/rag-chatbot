// data/chunks.json의 조각마다 벡터(숫자 768개)를 만들어 public/docs.json에 저장한다.
// 사용법 (rag-chatbot 폴더에서): node scripts/embed-docs.mjs
//
// 브라우저(src/rag.ts)가 질문을 벡터로 바꾸는 방법과 똑같이 만든다.
//   - 모델 파일: embeddinggemma-300m-ONNX의 model_no_gather_q4 (처음 한 번 .model 폴더로 받음, 약 200MB)
//   - 토크나이저: transformers.js AutoTokenizer
//   - mean pooling(attention_mask 기준) + L2 정규화
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { AutoTokenizer } from "@huggingface/transformers";
import * as ort from "onnxruntime-node";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX";
const HF_ONNX = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx`;
const MODEL_DIR = join(ROOT, ".model");
const FILES = ["model_no_gather_q4.onnx", "model_no_gather_q4.onnx_data"];

async function download(file) {
  const dest = join(MODEL_DIR, file);
  if (existsSync(dest)) return;
  console.log(`모델 파일 받는 중: ${file}`);
  const res = await fetch(`${HF_ONNX}/${file}`);
  if (!res.ok) throw new Error(`받기 실패 (${res.status}): ${file}`);
  writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
}

mkdirSync(MODEL_DIR, { recursive: true });
for (const f of FILES) await download(f);
const session = await ort.InferenceSession.create(join(MODEL_DIR, FILES[0]));
const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);

// 브라우저 rag.ts의 embed()와 같은 계산
async function embed(text) {
  const { input_ids, attention_mask } = await tokenizer(text);
  const feeds = {
    input_ids: new ort.Tensor("int64", input_ids.data, input_ids.dims),
    attention_mask: new ort.Tensor("int64", attention_mask.data, attention_mask.dims),
  };
  const hs = (await session.run(feeds)).last_hidden_state;
  const [, seq, hid] = hs.dims;
  const am = attention_mask.data;
  const acc = new Float64Array(hid);
  let cnt = 0;
  for (let s = 0; s < seq; s++) {
    const w = Number(am[s]);
    cnt += w;
    if (!w) continue;
    for (let h = 0; h < hid; h++) acc[h] += hs.data[s * hid + h];
  }
  let norm = 0;
  for (let h = 0; h < hid; h++) { acc[h] /= cnt; norm += acc[h] * acc[h]; }
  norm = Math.sqrt(norm);
  return Array.from(acc, (x) => Math.round((x / norm) * 1e6) / 1e6);
}

const chunks = JSON.parse(readFileSync(join(ROOT, "data", "chunks.json"), "utf8"));
const out = [];
for (const c of chunks) {
  // 조각 제목(section)과 본문을 함께 벡터로 만든다
  const vector = await embed(`${c.section}\n${c.text}`);
  if (vector.length !== 768) throw new Error(`${c.id}: 벡터가 ${vector.length}차원입니다`);
  out.push({ ...c, vector });
  process.stdout.write(`\r${out.length}/${chunks.length} ${c.id}   `);
}
writeFileSync(join(ROOT, "public", "docs.json"), JSON.stringify(out));
console.log(`\n저장: public/docs.json — ${out.length}조각 × 768차원`);
