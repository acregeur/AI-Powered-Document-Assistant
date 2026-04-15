const chatThread = document.getElementById("chatThread");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const folderPathInput = document.getElementById("folderPath");
const ingestButton = document.getElementById("ingestButton");
const ingestStatus = document.getElementById("ingestStatus");
const sendButton = document.getElementById("sendButton");
const healthBadge = document.getElementById("healthBadge");
const messageTemplate = document.getElementById("messageTemplate");
let minSourceSimilarityScore = 0.35;

function getVisibleSources(sources = []) {
  return sources.filter(
    (source) => (source.similarity_score ?? 0) >= minSourceSimilarityScore
  );
}

function autoResizeTextarea() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 220)}px`;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  ingestButton.disabled = isBusy;
  sendButton.textContent = isBusy ? "Thinking..." : "Send";
}

function appendMessage({ role, label, text, sources = [] }) {
  const fragment = messageTemplate.content.cloneNode(true);
  const message = fragment.querySelector(".message");
  const avatar = fragment.querySelector(".avatar");
  const messageLabel = fragment.querySelector(".message-label");
  const messageBody = fragment.querySelector(".message-body");
  const sourcesContainer = fragment.querySelector(".sources");

  message.classList.add(role);
  avatar.textContent = role === "user" ? "You" : "AI";
  messageLabel.textContent = label;
  messageBody.textContent = text;

  if (sources.length > 0) {
    sources.forEach((source) => {
      const chip = document.createElement("span");
      chip.className = "source-chip mono";
      const page = source.page_number ? ` • p.${source.page_number}` : "";
      chip.textContent = `${source.filename}${page}`;
      sourcesContainer.appendChild(chip);
    });
  } else {
    sourcesContainer.remove();
  }

  chatThread.appendChild(fragment);
  chatThread.scrollTop = chatThread.scrollHeight;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("API unavailable");
    }
    const payload = await response.json();
    minSourceSimilarityScore =
      payload.min_source_similarity_score ?? minSourceSimilarityScore;
    healthBadge.textContent = `${payload.status.toUpperCase()} • ${payload.version}`;
  } catch (error) {
    healthBadge.textContent = "API unavailable";
  }
}

async function ingestFolder() {
  const folder_path = folderPathInput.value.trim();

  if (!folder_path) {
    ingestStatus.textContent = "Enter a local folder path before indexing.";
    return;
  }

  ingestStatus.textContent = "Building knowledge base...";
  setBusy(true);

  try {
    const response = await fetch("/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_path }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Ingestion failed.");
    }

    ingestStatus.textContent = `Indexed ${payload.indexed_files} file(s) and ${payload.indexed_chunks} chunk(s).`;
    appendMessage({
      role: "assistant",
      label: "System",
      text: `Knowledge base refreshed from ${payload.folder_path}.`,
    });
  } catch (error) {
    ingestStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

async function submitQuestion(event) {
  event.preventDefault();

  const question = questionInput.value.trim();

  if (!question) {
    return;
  }

  appendMessage({ role: "user", label: "You", text: question });
  questionInput.value = "";
  autoResizeTextarea();
  setBusy(true);

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Query failed.");
    }

    appendMessage({
      role: "assistant",
      label: "Assistant",
      text: payload.answer,
      sources: getVisibleSources(payload.sources || []),
    });
  } catch (error) {
    appendMessage({
      role: "assistant",
      label: "Assistant",
      text: error.message,
    });
  } finally {
    setBusy(false);
    questionInput.focus();
  }
}

questionInput.addEventListener("input", autoResizeTextarea);
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

ingestButton.addEventListener("click", ingestFolder);
chatForm.addEventListener("submit", submitQuestion);

autoResizeTextarea();
checkHealth();
