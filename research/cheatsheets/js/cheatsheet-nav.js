/**
 * Renders grouped side nav on all cheatsheet pages and marks the active link.
 * Nav groups are collapsible; state persists in localStorage.
 */
(function () {
  const NAV_COLLAPSE_STORAGE_KEY = "qlab-cheatsheet-nav-collapsed";

  /** @type {{ label: string, items: { href: string, label: string }[] }[]} */
  const NAV_GROUPS = [
    {
      label: "Training & optimization",
      items: [
        { href: "index.html", label: "Activation functions" },
        { href: "loss-functions.html", label: "Loss functions" },
        { href: "generalization.html", label: "Generalization & regularization" },
        { href: "metrics.html", label: "Metrics & evaluation" },
        { href: "training-data.html", label: "Data for training" },
        { href: "training-loop.html", label: "Training loop & schedules" },
        { href: "optimizers.html", label: "Optimizers" },
        { href: "failure-modes.html", label: "Failure modes" },
        { href: "troubleshooting.html", label: "How to troubleshoot" },
        { href: "cloud-deep-learning.html", label: "Cloud deep learning" },
      ],
    },
    {
      label: "Testing",
      items: [
        { href: "testing-concepts.html", label: "Concepts & objectives" },
        { href: "testing-strategies.html", label: "Strategies" },
        { href: "testing-python.html", label: "Python, pytest & mocks" },
        { href: "testing-ml.html", label: "ML & LLM systems" },
        { href: "testing-llm-judge.html", label: "LLM as a judge" },
        { href: "testing-performance.html", label: "Performance & load" },
      ],
    },
    {
      label: "DevOps",
      items: [
        { href: "devops-overview.html", label: "Overview" },
        { href: "devops-containers.html", label: "Containers" },
        { href: "devops-kubernetes.html", label: "Kubernetes" },
        { href: "devops-cicd.html", label: "CI/CD" },
        { href: "devops-infrastructure.html", label: "Infrastructure" },
        { href: "devops-observability.html", label: "Observability" },
      ],
    },
    {
      label: "MLOps",
      items: [
        { href: "mlops-lifecycle.html", label: "Lifecycle" },
        { href: "mlops-pipelines.html", label: "Pipelines & CI" },
        { href: "mlops-registry.html", label: "Registry & promotion" },
        { href: "mlops-monitoring.html", label: "Monitoring & drift" },
        { href: "mlops-governance.html", label: "Governance & lineage" },
      ],
    },
    {
      label: "Transformers",
      items: [
        { href: "transformers-core.html", label: "Core & attention" },
        { href: "transformers-models.html", label: "Model lineage" },
        { href: "transformers-efficiency.html", label: "Efficiency" },
        { href: "transformers-serving.html", label: "Serving & hosting" },
        { href: "transformers-serving-ray-vllm.html", label: "Ray Serve + vLLM prod" },
        { href: "transformers-serving-autoscaling.html", label: "LLM autoscaling (full picture)" },
        { href: "transformers-compression.html", label: "Compression" },
        { href: "transformers-peft.html", label: "PEFT & adapters" },
        { href: "transformers-vision.html", label: "Vision" },
        { href: "transformers-image-gen.html", label: "Image generation" },
        { href: "transformers-audio.html", label: "Audio & speech" },
      ],
    },
    {
      label: "Databases",
      items: [
        { href: "databases-by-type.html", label: "By type" },
        { href: "cloud-databases.html", label: "Cloud" },
      ],
    },
    {
      label: "Security",
      items: [
        { href: "security-access-control.html", label: "Concepts" },
        { href: "security-implementation.html", label: "Implement & deploy" },
        { href: "security-genai-threats.html", label: "GenAI threats" },
        { href: "security-genai-limits.html", label: "GenAI rate limits" },
      ],
    },
    {
      label: "RAG",
      items: [
        { href: "rag-retrieval.html", label: "Retrieval overview" },
        { href: "rag-fusion-ranking.html", label: "Fusion & ranking" },
        { href: "rag-representations.html", label: "Representations" },
        { href: "rag-ann.html", label: "ANN indexes" },
        { href: "rag-chunking.html", label: "Chunking" },
        { href: "rag-ingest-pipeline.html", label: "Ingest (ETL)" },
        { href: "rag-systems.html", label: "RAG systems" },
        { href: "hallucination-diagnosis.html", label: "Hallucination (diagnosis)" },
        { href: "rag-evaluation.html", label: "Evaluation" },
      ],
    },
    {
      label: "Agents",
      items: [
        { href: "agents-llms.html", label: "LLM comparison" },
        { href: "agents-model-eval.html", label: "Local model eval" },
        { href: "agents-patterns.html", label: "Control patterns" },
        { href: "agents-memory.html", label: "Memory & state" },
        { href: "agents-learning.html", label: "Learning & improvement" },
        { href: "agents-trust-policy.html", label: "Trust & use policy" },
        { href: "agents-grounding.html", label: "Grounding & hallucination" },
        { href: "agents-system-security.html", label: "System & security" },
        { href: "agents-reliability.html", label: "Reliability & ops" },
      ],
    },
    {
      label: "Interview prep",
      items: [{ href: "ai-engineer-interview.html", label: "AI Engineer" }],
    },
  ];

  /** @returns {Set<string>} */
  function loadCollapsedLabels() {
    try {
      const raw = localStorage.getItem(NAV_COLLAPSE_STORAGE_KEY);
      if (!raw) {
        return new Set();
      }
      const parsed = JSON.parse(raw);
      if (!parsed || !Array.isArray(parsed.labels)) {
        return new Set();
      }
      return new Set(parsed.labels.filter((label) => typeof label === "string"));
    } catch {
      return new Set();
    }
  }

  /** @param {Set<string>} collapsed */
  function saveCollapsedLabels(collapsed) {
    localStorage.setItem(
      NAV_COLLAPSE_STORAGE_KEY,
      JSON.stringify({ labels: [...collapsed] }),
    );
  }

  /** @param {HTMLElement} group */
  function setGroupCollapsed(group, collapsed) {
    const toggle = group.querySelector(".nav-group-toggle");
    const list = group.querySelector(".nav-list");
    group.classList.toggle("is-collapsed", collapsed);
    if (toggle) {
      toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    }
    if (list) {
      list.hidden = collapsed;
    }
  }

  /** @param {Set<string>} collapsed */
  function applyCollapsedState(collapsed) {
    document.querySelectorAll(".nav-group[data-nav-label]").forEach((group) => {
      if (!(group instanceof HTMLElement)) {
        return;
      }
      const label = group.dataset.navLabel;
      if (!label) {
        return;
      }
      setGroupCollapsed(group, collapsed.has(label));
    });
  }

  /** @returns {Set<string>} */
  function readCollapsedFromDom() {
    const collapsed = new Set();
    document.querySelectorAll(".nav-group[data-nav-label]").forEach((group) => {
      if (!(group instanceof HTMLElement)) {
        return;
      }
      const label = group.dataset.navLabel;
      if (label && group.classList.contains("is-collapsed")) {
        collapsed.add(label);
      }
    });
    return collapsed;
  }

  function renderNav() {
    const mount = document.querySelector("[data-cheatsheet-nav]");
    if (!mount) {
      return;
    }

    const frag = document.createDocumentFragment();

    const tools = document.createElement("div");
    tools.className = "nav-tools";
    tools.setAttribute("role", "toolbar");
    tools.setAttribute("aria-label", "Navigation sections");

    const expandAllBtn = document.createElement("button");
    expandAllBtn.type = "button";
    expandAllBtn.className = "nav-tools-btn";
    expandAllBtn.dataset.navAction = "expand-all";
    expandAllBtn.textContent = "Expand all";

    const collapseAllBtn = document.createElement("button");
    collapseAllBtn.type = "button";
    collapseAllBtn.className = "nav-tools-btn";
    collapseAllBtn.dataset.navAction = "collapse-all";
    collapseAllBtn.textContent = "Collapse all";

    const sep = document.createElement("span");
    sep.className = "nav-tools-sep";
    sep.setAttribute("aria-hidden", "true");
    sep.textContent = "·";

    tools.appendChild(expandAllBtn);
    tools.appendChild(sep);
    tools.appendChild(collapseAllBtn);
    frag.appendChild(tools);

    NAV_GROUPS.forEach((groupDef, index) => {
      const group = document.createElement("section");
      group.className = "nav-group";
      group.dataset.navLabel = groupDef.label;
      if (index === 0) {
        group.classList.add("nav-group-first");
      }

      const listId = `nav-group-list-${index}`;

      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "nav-group-toggle";
      toggle.setAttribute("aria-expanded", "true");
      toggle.setAttribute("aria-controls", listId);

      const labelSpan = document.createElement("span");
      labelSpan.className = "nav-group-label";
      labelSpan.textContent = groupDef.label;

      const chevron = document.createElement("span");
      chevron.className = "nav-group-chevron";
      chevron.setAttribute("aria-hidden", "true");

      toggle.appendChild(labelSpan);
      toggle.appendChild(chevron);
      group.appendChild(toggle);

      const ul = document.createElement("ul");
      ul.id = listId;
      ul.className = "nav-list nav-list-nested";
      groupDef.items.forEach((item) => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = item.href;
        a.textContent = item.label;
        li.appendChild(a);
        ul.appendChild(li);
      });
      group.appendChild(ul);
      frag.appendChild(group);
    });

    mount.replaceChildren(frag);
  }

  function markActiveLink() {
    const path = window.location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll(".sheet-nav a[href]").forEach((link) => {
      const href = link.getAttribute("href");
      if (href === path || (path === "" && href === "index.html")) {
        link.classList.add("is-active");
        link.setAttribute("aria-current", "page");
        const group = link.closest(".nav-group");
        if (group instanceof HTMLElement) {
          group.classList.add("nav-group-active");
          setGroupCollapsed(group, false);
          const label = group.dataset.navLabel;
          if (label) {
            const collapsed = loadCollapsedLabels();
            collapsed.delete(label);
            saveCollapsedLabels(collapsed);
          }
        }
      }
    });
  }

  function bindNavInteractions() {
    const mount = document.querySelector("[data-cheatsheet-nav]");
    if (!mount) {
      return;
    }

    mount.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }

      const actionBtn = target.closest("[data-nav-action]");
      if (actionBtn instanceof HTMLElement) {
        const action = actionBtn.dataset.navAction;
        if (action === "expand-all") {
          document.querySelectorAll(".nav-group[data-nav-label]").forEach((group) => {
            if (group instanceof HTMLElement) {
              setGroupCollapsed(group, false);
            }
          });
          saveCollapsedLabels(new Set());
        } else if (action === "collapse-all") {
          const allLabels = new Set(NAV_GROUPS.map((g) => g.label));
          applyCollapsedState(allLabels);
          saveCollapsedLabels(allLabels);
        }
        return;
      }

      const toggle = target.closest(".nav-group-toggle");
      if (toggle instanceof HTMLElement) {
        const group = toggle.closest(".nav-group");
        if (!(group instanceof HTMLElement)) {
          return;
        }
        const willCollapse = !group.classList.contains("is-collapsed");
        setGroupCollapsed(group, willCollapse);
        saveCollapsedLabels(readCollapsedFromDom());
      }
    });
  }

  function init() {
    renderNav();
    applyCollapsedState(loadCollapsedLabels());
    markActiveLink();
    bindNavInteractions();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
