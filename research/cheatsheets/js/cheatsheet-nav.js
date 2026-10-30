/**
 * Renders grouped side nav on all cheatsheet pages and marks the active link.
 */
(function () {
  /** @type {{ label: string, items: { href: string, label: string }[] }[]} */
  const NAV_GROUPS = [
    {
      label: "Training & optimization",
      items: [
        { href: "index.html", label: "Activation functions" },
        { href: "loss-functions.html", label: "Loss functions" },
        { href: "optimizers.html", label: "Optimizers" },
        { href: "failure-modes.html", label: "Failure modes" },
      ],
    },
    {
      label: "Transformers",
      items: [
        { href: "transformers-core.html", label: "Core & attention" },
        { href: "transformers-efficiency.html", label: "Efficiency" },
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
      ],
    },
    {
      label: "RAG",
      items: [
        { href: "rag-retrieval.html", label: "Retrieval overview" },
        { href: "rag-representations.html", label: "Representations" },
        { href: "rag-ann.html", label: "ANN indexes" },
        { href: "rag-chunking.html", label: "Chunking" },
        { href: "rag-systems.html", label: "RAG systems" },
      ],
    },
    {
      label: "Interview prep",
      items: [{ href: "ai-engineer-interview.html", label: "AI Engineer" }],
    },
  ];

  function renderNav() {
    const mount = document.querySelector("[data-cheatsheet-nav]");
    if (!mount) {
      return;
    }

    const frag = document.createDocumentFragment();

    NAV_GROUPS.forEach((groupDef, index) => {
      const group = document.createElement("section");
      group.className = "nav-group";
      if (index === 0) {
        group.classList.add("nav-group-first");
      }

      const heading = document.createElement("h2");
      heading.className = "nav-group-label";
      heading.textContent = groupDef.label;
      group.appendChild(heading);

      const ul = document.createElement("ul");
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
        if (group) {
          group.classList.add("nav-group-active");
        }
      }
    });
  }

  function init() {
    renderNav();
    markActiveLink();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
