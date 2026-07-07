import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import torch


ACTIONS = ["up", "down", "left", "right", "touch"]
PUNCTUATION = ".,;:!?()[]{}<>\"'`~@#$%^&*-_=+\\/|"


def read_memory(memory_path):
    uri = memory_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conversation_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(conversations)")
        }
        source_select = "source" if "source" in conversation_columns else "'unknown' AS source"
        conversations = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT question, answer, timestamp, {source_select}
                FROM conversations
                ORDER BY timestamp ASC
                """
            )
        ]
        episodes = [
            dict(row)
            for row in conn.execute(
                """
                SELECT observation, action, outcome, surprise_level, timestamp
                FROM episodes
                ORDER BY timestamp ASC
                """
            )
        ]
    return conversations, episodes


def normalize_sequence(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def normalize_number(value, default=0):
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_checkpoint(checkpoint_path):
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(state, dict):
        state = {}

    vocabulary = [str(item) for item in normalize_sequence(state.get("language_vocabulary"))]
    history = [str(item) for item in normalize_sequence(state.get("language_history"))]

    emotional_state = state.get("emotional_state") or {}
    if not isinstance(emotional_state, dict):
        emotional_state = {}
    emotional_state = {
        key: normalize_number(emotional_state.get(key))
        for key in ["curiosity", "fear", "confidence", "confusion"]
    }

    action_counts = [int(normalize_number(item)) for item in normalize_sequence(state.get("action_counts"))]
    action_counts = (action_counts + [0] * len(ACTIONS))[: len(ACTIONS)]

    return {
        "language_vocabulary": vocabulary,
        "language_history": history,
        "step_count": int(normalize_number(state.get("step_count"))),
        "emotional_state": emotional_state,
        "action_counts": action_counts,
    }


def normalize_source(value):
    source = str(value or "").strip()
    if source in {"human_taught", "agent_generated"}:
        return source
    return "unknown"


def conversation_source_stats(conversations):
    labels = ["human_taught", "agent_generated", "unknown"]
    total = len(conversations)
    counts = {label: 0 for label in labels}
    for conversation in conversations:
        counts[normalize_source(conversation.get("source"))] += 1
    return [
        {
            "source": label,
            "count": counts[label],
            "percentage": (counts[label] / total * 100) if total else 0,
        }
        for label in labels
    ]


def action_distribution(action_counts):
    total = sum(action_counts)
    return [
        {
            "action": action,
            "count": count,
            "percentage": (count / total * 100) if total else 0,
        }
        for action, count in zip(ACTIONS, action_counts)
    ]


def word_frequencies(conversations, source=None):
    counts = {}
    for conversation in conversations:
        if source is not None and normalize_source(conversation.get("source")) != source:
            continue
        for raw_word in str(conversation.get("answer", "")).lower().split():
            word = raw_word.strip(PUNCTUATION)
            if len(word) < 3:
                continue
            counts[word] = counts.get(word, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:30]


def sample_points(items, max_points=200):
    if len(items) <= max_points:
        return items
    step = (len(items) - 1) / (max_points - 1)
    return [items[round(index * step)] for index in range(max_points)]


def build_report(data):
    payload = (
        json.dumps(data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Analytics Report</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1d232b;
      --muted: #667085;
      --line: #d9dee7;
      --blue: #2f6fed;
      --green: #159a74;
      --gold: #c58a12;
      --red: #d94841;
      --violet: #7755cc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
    }}
    main {{
      width: min(1440px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    header {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: 0;
    }}
    .stamp {{
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .stat, .chart-panel, .table-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }}
    .stat {{
      padding: 14px 16px;
      min-height: 82px;
    }}
    .stat-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .stat-value {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
      line-height: 1;
    }}
    .stat-breakdown, .value-list {{
      display: grid;
      gap: 6px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.25;
    }}
    .stat-breakdown div, .value-list div, .mini-table-row {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }}
    .stat-breakdown span:last-child, .value-list span:last-child, .mini-table-row span:last-child {{
      color: var(--ink);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .chart-panel {{
      padding: 16px;
      min-height: 330px;
    }}
    .chart-title {{
      margin: 0 0 12px;
      font-size: 16px;
      line-height: 1.25;
    }}
    .chart-frame {{
      position: relative;
      height: 270px;
    }}
    .chart-with-table {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 210px;
      gap: 14px;
      align-items: stretch;
    }}
    .chart-with-table .chart-frame {{
      min-width: 0;
    }}
    .mini-table {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      align-self: center;
      font-size: 12px;
    }}
    .mini-table-row {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
    }}
    .mini-table-row:last-child {{
      border-bottom: 0;
    }}
    .mini-table-head {{
      background: #eef1f6;
      color: #344054;
      font-weight: 700;
    }}
    .table-panel {{
      margin-top: 16px;
      overflow: hidden;
    }}
    .table-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }}
    .table-head h2 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
    }}
    .table-wrap {{
      max-height: 560px;
      overflow: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
      overflow-wrap: anywhere;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #eef1f6;
      color: #344054;
      font-size: 12px;
    }}
    th:nth-child(1), td:nth-child(1) {{ width: 15%; }}
    th:nth-child(2), td:nth-child(2) {{ width: 32%; }}
    th:nth-child(3), td:nth-child(3) {{ width: 53%; }}
    @media (max-width: 900px) {{
      header {{ align-items: start; flex-direction: column; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
      .chart-with-table {{ grid-template-columns: 1fr; }}
      .stamp {{ white-space: normal; }}
    }}
    @media (max-width: 560px) {{
      main {{ width: min(100vw - 20px, 1440px); padding-top: 18px; }}
      .stats {{ grid-template-columns: 1fr; }}
      th:nth-child(1), td:nth-child(1) {{ width: 22%; }}
      th:nth-child(2), td:nth-child(2) {{ width: 34%; }}
      th:nth-child(3), td:nth-child(3) {{ width: 44%; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Agent Analytics Report</h1>
      </div>
      <div class="stamp" id="generated-at"></div>
    </header>

    <section class="stats">
      <div class="stat"><div class="stat-label">Total steps</div><div class="stat-value" id="total-steps">0</div></div>
      <div class="stat"><div class="stat-label">Vocabulary size</div><div class="stat-value" id="vocabulary-size">0</div></div>
      <div class="stat"><div class="stat-label">Conversations</div><div class="stat-value" id="conversation-count">0</div></div>
      <div class="stat"><div class="stat-label">Episodes</div><div class="stat-value" id="episode-count">0</div></div>
      <div class="stat">
        <div class="stat-label">Conversations by source</div>
        <div class="stat-breakdown" id="conversation-source-breakdown"></div>
      </div>
    </section>

    <section class="grid">
      <article class="chart-panel">
        <h2 class="chart-title">Top Agent Answer Words - Agent Generated Only</h2>
        <div class="chart-frame"><canvas id="agentWordChart"></canvas></div>
      </article>
      <article class="chart-panel">
        <h2 class="chart-title">Top Agent Answer Words - All Sources</h2>
        <div class="chart-frame"><canvas id="wordChart"></canvas></div>
      </article>
      <article class="chart-panel">
        <h2 class="chart-title">Action Distribution</h2>
        <div class="chart-with-table">
          <div class="chart-frame"><canvas id="actionChart"></canvas></div>
          <div class="mini-table" id="action-distribution-table"></div>
        </div>
      </article>
      <article class="chart-panel">
        <h2 class="chart-title">Surprise Over Time</h2>
        <div class="chart-frame"><canvas id="surpriseChart"></canvas></div>
      </article>
      <article class="chart-panel">
        <h2 class="chart-title">Current Emotional State</h2>
        <div class="chart-frame"><canvas id="emotionChart"></canvas></div>
        <div class="value-list" id="emotion-values"></div>
      </article>
    </section>

    <section class="table-panel">
      <div class="table-head">
        <h2>Conversation History</h2>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Timestamp</th><th>Question</th><th>Answer</th></tr>
          </thead>
          <tbody id="conversation-rows"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const reportData = {payload};
    const numberFormat = new Intl.NumberFormat();

    function setText(id, value) {{
      document.getElementById(id).textContent = value;
    }}

    function formatTimestamp(value) {{
      const numeric = Number(value);
      if (Number.isFinite(numeric)) {{
        const millis = numeric > 100000000000 ? numeric : numeric * 1000;
        const date = new Date(millis);
        if (!Number.isNaN(date.getTime())) {{
          return date.toLocaleString();
        }}
      }}
      return String(value ?? "");
    }}

    setText("generated-at", "Generated " + new Date(reportData.generated_at).toLocaleString());
    setText("total-steps", numberFormat.format(reportData.summary.total_steps));
    setText("vocabulary-size", numberFormat.format(reportData.summary.vocabulary_size));
    setText("conversation-count", numberFormat.format(reportData.summary.total_conversations));
    setText("episode-count", numberFormat.format(reportData.summary.total_episodes));

    function formatPercent(value) {{
      return Number(value || 0).toFixed(2) + "%";
    }}

    const sourceBreakdown = document.getElementById("conversation-source-breakdown");
    for (const item of reportData.conversation_sources) {{
      const row = document.createElement("div");
      const label = document.createElement("span");
      const value = document.createElement("span");
      label.textContent = item.source;
      value.textContent = numberFormat.format(item.count) + " (" + formatPercent(item.percentage) + ")";
      row.appendChild(label);
      row.appendChild(value);
      sourceBreakdown.appendChild(row);
    }}

    const commonOptions = {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ labels: {{ boxWidth: 14, color: "#344054" }} }}
      }},
      scales: {{
        x: {{ ticks: {{ color: "#667085", maxRotation: 0, autoSkip: true }}, grid: {{ color: "#edf0f5" }} }},
        y: {{ ticks: {{ color: "#667085" }}, grid: {{ color: "#edf0f5" }} }}
      }}
    }};

    new Chart(document.getElementById("agentWordChart"), {{
      type: "bar",
      data: {{
        labels: reportData.agent_generated_word_frequency.map(item => item.word),
        datasets: [{{
          label: "Agent-generated count",
          data: reportData.agent_generated_word_frequency.map(item => item.count),
          backgroundColor: "#159a74"
        }}]
      }},
      options: commonOptions
    }});

    new Chart(document.getElementById("wordChart"), {{
      type: "bar",
      data: {{
        labels: reportData.word_frequency.map(item => item.word),
        datasets: [{{
          label: "Count",
          data: reportData.word_frequency.map(item => item.count),
          backgroundColor: "#2f6fed"
        }}]
      }},
      options: commonOptions
    }});

    new Chart(document.getElementById("actionChart"), {{
      type: "pie",
      data: {{
        labels: reportData.action_counts.labels,
        datasets: [{{
          data: reportData.action_counts.values,
          backgroundColor: ["#2f6fed", "#159a74", "#c58a12", "#d94841", "#7755cc"]
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ position: "right", labels: {{ boxWidth: 14, color: "#344054" }} }} }}
      }}
    }});

    const actionTable = document.getElementById("action-distribution-table");
    const actionHead = document.createElement("div");
    actionHead.className = "mini-table-row mini-table-head";
    actionHead.innerHTML = "<span>Action</span><span>Percent</span>";
    actionTable.appendChild(actionHead);
    for (const item of reportData.action_distribution) {{
      const row = document.createElement("div");
      row.className = "mini-table-row";
      const action = document.createElement("span");
      const value = document.createElement("span");
      action.textContent = item.action + " (" + numberFormat.format(item.count) + ")";
      value.textContent = formatPercent(item.percentage);
      row.appendChild(action);
      row.appendChild(value);
      actionTable.appendChild(row);
    }}

    new Chart(document.getElementById("surpriseChart"), {{
      type: "line",
      data: {{
        labels: reportData.surprise_series.map(item => formatTimestamp(item.timestamp)),
        datasets: [{{
          label: "Surprise level",
          data: reportData.surprise_series.map(item => item.surprise_level),
          borderColor: "#d94841",
          backgroundColor: "rgba(217, 72, 65, 0.16)",
          pointRadius: 1.5,
          tension: 0.25,
          fill: true
        }}]
      }},
      options: commonOptions
    }});

    new Chart(document.getElementById("emotionChart"), {{
      type: "bar",
      data: {{
        labels: reportData.emotional_state.labels,
        datasets: [{{
          label: "Value",
          data: reportData.emotional_state.values,
          backgroundColor: ["#159a74", "#d94841", "#2f6fed", "#c58a12"]
        }}]
      }},
      options: {{
        ...commonOptions,
        indexAxis: "y",
        scales: {{
          x: {{ beginAtZero: true, ticks: {{ color: "#667085" }}, grid: {{ color: "#edf0f5" }} }},
          y: {{ ticks: {{ color: "#667085" }}, grid: {{ display: false }} }}
        }}
      }}
    }});

    const emotionValues = document.getElementById("emotion-values");
    reportData.emotional_state.labels.forEach((label, index) => {{
      const row = document.createElement("div");
      const name = document.createElement("span");
      const value = document.createElement("span");
      name.textContent = label + ":";
      value.textContent = Number(reportData.emotional_state.values[index] || 0).toFixed(3);
      row.appendChild(name);
      row.appendChild(value);
      emotionValues.appendChild(row);
    }});

    const rows = document.getElementById("conversation-rows");
    for (const item of reportData.conversations) {{
      const row = document.createElement("tr");
      for (const value of [formatTimestamp(item.timestamp), item.question, item.answer]) {{
        const cell = document.createElement("td");
        cell.textContent = value ?? "";
        row.appendChild(cell);
      }}
      rows.appendChild(row);
    }}
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate a standalone HTML analytics report.")
    parser.add_argument("--memory", required=True, type=Path, help="Path to episodic_memory.sqlite3")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to agent_state.pt")
    args = parser.parse_args()

    memory_path = args.memory.expanduser()
    checkpoint_path = args.checkpoint.expanduser()
    if not memory_path.is_file():
        raise FileNotFoundError(f"Memory database not found: {memory_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    conversations, episodes = read_memory(memory_path)
    checkpoint = load_checkpoint(checkpoint_path)
    surprise_points = sample_points(
        [
            {
                "timestamp": episode["timestamp"],
                "surprise_level": normalize_number(episode["surprise_level"]),
            }
            for episode in episodes
        ]
    )
    word_counts = word_frequencies(conversations)
    agent_generated_word_counts = word_frequencies(conversations, source="agent_generated")
    emotion_labels = ["curiosity", "fear", "confidence", "confusion"]

    data = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_steps": checkpoint["step_count"],
            "vocabulary_size": len(checkpoint["language_vocabulary"]),
            "total_conversations": len(conversations),
            "total_episodes": len(episodes),
        },
        "conversation_sources": conversation_source_stats(conversations),
        "agent_generated_word_frequency": [
            {"word": word, "count": count}
            for word, count in agent_generated_word_counts
        ],
        "word_frequency": [
            {"word": word, "count": count}
            for word, count in word_counts
        ],
        "action_counts": {
            "labels": ACTIONS,
            "values": checkpoint["action_counts"],
        },
        "action_distribution": action_distribution(checkpoint["action_counts"]),
        "surprise_series": surprise_points,
        "emotional_state": {
            "labels": emotion_labels,
            "values": [checkpoint["emotional_state"].get(label, 0) for label in emotion_labels],
        },
        "language_history": checkpoint["language_history"],
        "conversations": conversations,
    }

    report_path = Path(__file__).resolve().parent / "report.html"
    report_path.write_text(build_report(data), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
