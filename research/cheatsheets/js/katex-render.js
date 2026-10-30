/**
 * Render LaTeX in cheatsheet pages. Requires KaTeX + auto-render loaded before this script.
 * Delimiters: $…$ inline, $$…$$ display (use in .formula blocks).
 */
(function () {
  function render() {
    if (typeof renderMathInElement !== "function") {
      return;
    }
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
      strict: "ignore",
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render);
  } else {
    render();
  }
})();
