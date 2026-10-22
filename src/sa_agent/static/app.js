"use strict";

const form = document.querySelector("#research-form");
const question = document.querySelector("#question");
const submit = document.querySelector("#submit");
const dialog = document.querySelector("#evidence-dialog");
const statusLabels = {success: "成功", failed: "失败", skipped: "跳过", pending: "待执行", running: "执行中"};
const outcomeLabels = {no_documents: "未检索到片段", no_facts: "有片段，未提取出事实", no_verified_facts: "没有通过核验的事实", execution_failed: "执行失败", evidence_available: "证据可用"};
let busy = false;
let documentBusy = false;
let evidenceMap = new Map();

/** 创建文本节点，不将模型返回内容作为 HTML 执行。 */
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

/** 设置结果区域标题及辅助阅读器通知。 */
function setStatus(text, state) {
  const badge = document.querySelector("#status");
  badge.textContent = text;
  badge.dataset.state = state;
  document.querySelector("#status-message").textContent = text;
}

/** 控制提交和输入状态，防止同一次研究被重复发起。 */
function setBusy(value) {
  busy = value;
  updateControls();
  document.querySelector("#submit-label").textContent = value ? "研究中，请稍候" : "开始研究";
  document.querySelector("#loading").hidden = !value;
  document.querySelector(".content").setAttribute("aria-busy", String(value));
}

/** 导入与研究不同时提交，避免同页操作期间知识库发生变化。 */
function updateControls() {
  for (const control of document.querySelectorAll("#research-form textarea, #research-form select, #research-form button, .example, #upload-form input, #upload-form button, .retry-document")) control.disabled = busy || documentBusy;
}

/** 统一显示文件来源，PDF额外显示从1开始的页码。 */
function sourceLabel(item) {
  const page = Number.isInteger(item.page) ? ` · 第 ${item.page} 页` : "";
  return `${item.source}${page} · 片段 ${item.chunk_index ?? "未标注"}`;
}

/** 校验渲染所需字段，避免将异常响应误展示为研究成功。 */
function validateResult(data) {
  const report = data?.report;
  if (typeof data?.question !== "string" || typeof report?.summary !== "string"
      || !Array.isArray(report.findings) || !Array.isArray(report.gaps)
      || !Array.isArray(data.evidence) || !Array.isArray(data.tasks) || !Array.isArray(data.trace)) throw new Error("返回结果格式异常，请检查服务日志。");
  const ids = new Set();
  for (const item of data.evidence) {
    if (typeof item.id !== "string" || typeof item.source !== "string" || typeof item.evidence !== "string" || typeof item.fact !== "string" || ids.has(item.id)) throw new Error("返回的证据格式异常。");
    ids.add(item.id);
  }
  for (const finding of report.findings) {
    if (typeof finding.text !== "string" || !Array.isArray(finding.evidence_ids) || !finding.evidence_ids.length || finding.evidence_ids.some(id => !ids.has(id))) throw new Error("结论引用与证据不匹配。");
  }
  if (report.gaps.some(gap => typeof gap !== "string")) throw new Error("资料缺口格式异常。");
}

/** 根据引用编号展示来源、提取事实和原文证据。 */
function showEvidence(id) {
  const item = evidenceMap.get(id);
  if (!item) return;
  document.querySelector("#evidence-title").textContent = `证据 ${id}`;
  document.querySelector("#evidence-source").textContent = sourceLabel(item);
  document.querySelector("#evidence-fact").textContent = item.fact;
  document.querySelector("#evidence-text").textContent = item.evidence;
  dialog.showModal();
}

/** 生成可通过键盘操作的引用按钮。 */
function citation(id) {
  const button = element("button", `[${id}]`, "citation");
  button.type = "button";
  button.setAttribute("aria-label", `查看证据 ${id}`);
  button.addEventListener("click", () => showEvidence(id));
  return button;
}

/** 渲染结论、缺口及证据列表，资料不足也作为有效结果展示。 */
function renderReport(data) {
  evidenceMap = new Map(data.evidence.map(item => [item.id, item]));
  document.querySelector("#result-question").textContent = data.question;
  document.querySelector("#summary").textContent = data.report.summary;
  document.querySelector("#finding-count").textContent = `${data.report.findings.length} 条`;
  const findings = document.querySelector("#findings");
  findings.replaceChildren();
  for (const finding of data.report.findings) {
    const row = element("li");
    row.append(element("span", finding.text));
    for (const id of finding.evidence_ids) row.append(citation(id));
    findings.append(row);
  }
  if (!data.report.findings.length) findings.append(element("li", "暂无足够证据形成结论，请查看下方资料缺口。", "muted"));
  const gaps = document.querySelector("#gaps");
  gaps.replaceChildren(...data.report.gaps.map(text => element("li", text)));
  if (!data.report.gaps.length) gaps.append(element("li", "本次未列出特定缺口，不代表资料覆盖完整。"));
  const sources = document.querySelector("#evidence-list");
  sources.replaceChildren();
  for (const item of data.evidence) {
    const button = element("button", undefined, "evidence-item");
    button.type = "button";
    button.append(element("strong", `[${item.id}]`), element("span", item.fact), element("small", sourceLabel(item)));
    button.addEventListener("click", () => showEvidence(item.id));
    sources.append(button);
  }
  if (!data.evidence.length) sources.append(element("p", "本次暂无可引用证据。", "muted"));
}

/** 展示研究调用、任务计划和实际节点结果，不模拟实时执行。 */
function renderTrace(data) {
  const container = document.querySelector("#trace");
  container.replaceChildren();
  const taskMap = new Map(data.tasks.map(task => [task.id, task]));
  for (const entry of data.trace) {
    if (entry.step !== "research") {
      container.append(element("p", `调用被拒绝：${entry.reason ?? "请查看服务日志"}`, "muted"));
      continue;
    }
    const details = element("details", undefined, "trace-call");
    details.open = entry.call === 1;
    const title = element("summary", `第 ${entry.call} 次研究 · ${entry.plan?.tasks?.length ?? 0} 个任务`);
    title.append(element("span", entry.status === "completed" ? "已完成" : entry.status === "rejected" ? "已拒绝" : "执行失败", "trace-status"));
    details.append(title);
    for (const planned of entry.plan?.tasks ?? []) {
      const task = taskMap.get(planned.id);
      const info = entry.tasks?.find(item => item.id === planned.id);
      const block = element("div", undefined, "task-detail");
      block.append(element("strong", `${planned.id} · ${planned.goal}`), element("p", `检索查询：${planned.query}`, "query"));
      if (info) block.append(element("p", `${outcomeLabels[info.outcome] ?? info.outcome ?? ""} · 检索 ${info.counts?.retrieved ?? "—"} 片段 · 提取 ${info.counts?.extracted ?? "—"} 条 · 核验通过 ${info.counts?.passed ?? "—"} 条`, "outcome"));
      const nodes = element("div", undefined, "node-list");
      for (const [name, label] of [["retrieve", "检索"], ["extract", "提取"], ["verify", "核验"]]) {
        const state = task?.nodes?.[name];
        const node = element("span", `${label} · ${statusLabels[state] ?? "未执行"}`, "node");
        node.dataset.status = state ?? "skipped";
        nodes.append(node);
      }
      block.append(nodes);
      details.append(block);
    }
    container.append(details);
  }
}

/** 在成功响应后更新统计及全部结果区域。 */
function renderResult(data, seconds) {
  validateResult(data);
  const metrics = document.querySelector("#metrics");
  const count = data.trace.filter(entry => entry.step === "research").length;
  metrics.replaceChildren();
  for (const [value, label] of [[count, "次研究"], [data.tasks.length, "个任务"], [data.evidence.length, "条证据"], [seconds, "秒"]]) {
    const metric = element("span");
    metric.append(element("strong", String(value)), document.createTextNode(label));
    metrics.append(metric);
  }
  renderReport(data);
  renderTrace(data);
  document.querySelector("#results").hidden = false;
  setStatus(data.report.findings.length ? "研究完成" : "资料不足", "done");
}

/** 提交真实研究请求，覆盖等待、HTTP 错误和网络失败状态。 */
async function submitResearch(event) {
  event.preventDefault();
  if (busy || documentBusy) return;
  const text = question.value.trim();
  const error = document.querySelector("#error");
  error.hidden = true;
  if (!text) {
    error.textContent = "请输入有效的研究问题，不能仅包含空格。";
    error.hidden = false;
    question.focus();
    return;
  }
  const started = performance.now();
  document.querySelector("#empty").hidden = true;
  document.querySelector("#results").hidden = true;
  document.querySelector("#elapsed").textContent = "0";
  setBusy(true);
  setStatus("研究中", "busy");
  const timer = setInterval(() => { document.querySelector("#elapsed").textContent = String(Math.floor((performance.now() - started) / 1000)); }, 1000);
  try {
    const response = await fetch("/agent/research", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({question: text, k: Number(document.querySelector("#k").value)})});
    let data;
    try { data = await response.json(); } catch { throw new Error(`服务返回了无法解析的响应（HTTP ${response.status}），请检查服务是否正常。`); }
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `请求失败（HTTP ${response.status}），请检查输入或服务日志。`);
    renderResult(data, Math.round((performance.now() - started) / 1000));
  } catch (failure) {
    error.textContent = failure instanceof TypeError ? "无法连接研究服务，请确认服务仍在运行后重试。" : failure.message || "研究失败，请稍后重试。";
    error.hidden = false;
    document.querySelector("#results").hidden = true;
    document.querySelector("#empty").hidden = false;
    setStatus("研究失败", "error");
  } finally {
    clearInterval(timer);
    setBusy(false);
  }
}

/** 解析文档接口响应并提取可展示的服务端错误。 */
async function documentRequest(url, options) {
  let response;
  try { response = await fetch(url, options); } catch { throw new Error("无法连接文档服务，请确认服务仍在运行。"); }
  let data;
  try { data = await response.json(); } catch { throw new Error("文档服务响应格式异常，请检查服务日志。"); }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `文档请求失败（HTTP ${response.status}）`);
  return data;
}

/** 展示文档列表；片段数是服务端读取到的实际入库数量。 */
async function loadDocuments() {
  const container = document.querySelector("#document-list");
  try {
    const data = await documentRequest("/documents");
    if (!Array.isArray(data.documents)) throw new Error("文档列表格式异常。");
    container.replaceChildren();
    const labels = {imported: "已导入", processing: "处理中", failed: "导入失败", not_imported: "待导入"};
    for (const item of data.documents) {
      const row = element("article", undefined, "document-row");
      row.append(element("strong", item.filename), element("span", labels[item.status] ?? item.status, `document-status ${item.status}`));
      row.append(element("p", `${item.format} · ${Math.max(1, Math.ceil(item.size_bytes / 1024))} KB · ${item.chunks_count} 个已入库片段`, "muted"));
      if (item.message) row.append(element("p", item.message, "muted"));
      if (["failed", "not_imported"].includes(item.status)) {
        const retry = element("button", item.status === "failed" ? "重试导入" : "导入资料", "text-button retry-document");
        retry.type = "button";
        retry.addEventListener("click", () => retryDocument(item.filename));
        row.append(retry);
      }
      container.append(row);
    }
    if (!data.documents.length) container.append(element("p", "暂无文档，请先上传研究资料。", "muted"));
    updateControls();
  } catch (error) {
    container.replaceChildren(element("p", `列表读取失败：${error.message}`, "error"));
  }
}

/** 统一处理上传和重试的等待、错误与列表刷新。 */
async function importAction(action) {
  if (busy || documentBusy) return;
  documentBusy = true;
  updateControls();
  const message = document.querySelector("#document-message");
  const error = document.querySelector("#document-error");
  error.hidden = true;
  message.textContent = "正在上传或导入，请稍候；关闭窗口不会取消处理。";
  try {
    const result = await action();
    message.textContent = `${result.filename} 已导入，共 ${result.chunks_count} 个片段。可重新发起研究。`;
    document.querySelector("#document-file").value = "";
  } catch (failure) {
    message.textContent = "本次导入未完成。";
    error.textContent = failure.message;
    error.hidden = false;
  } finally {
    documentBusy = false;
    updateControls();
    await loadDocuments();
  }
}

/** 提交支持格式的单文件，multipart边界由浏览器自动生成。 */
async function uploadDocument(event) {
  event.preventDefault();
  const file = document.querySelector("#document-file").files[0];
  if (!file) return;
  const error = document.querySelector("#document-error");
  if (!/\.(txt|md|pdf|docx)$/i.test(file.name) || file.size === 0 || file.size > 10 * 1024 * 1024) {
    error.textContent = "请选择非空的 TXT、MD、DOCX 或文本 PDF，且文件不能超过10 MB。";
    error.hidden = false;
    return;
  }
  const body = new FormData();
  body.append("file", file);
  await importAction(() => documentRequest("/documents/upload", {method: "POST", body}));
}

/** 对失败或尚未导入的已有文件重试，不重新上传文件。 */
async function retryDocument(filename) {
  await importAction(() => documentRequest("/documents/retry", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({filename})}));
}

form.addEventListener("submit", submitResearch);
document.querySelector("#close-evidence").addEventListener("click", () => dialog.close());
document.querySelector("#open-documents").addEventListener("click", () => {
  document.querySelector("#documents-dialog").showModal();
  loadDocuments();
});
document.querySelector("#close-documents").addEventListener("click", () => document.querySelector("#documents-dialog").close());
document.querySelector("#refresh-documents").addEventListener("click", loadDocuments);
document.querySelector("#upload-form").addEventListener("submit", uploadDocument);
for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => {
    question.value = button.dataset.question;
    question.focus();
  });
}
