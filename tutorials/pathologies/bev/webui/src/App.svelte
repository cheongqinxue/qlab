<script>
  import { onMount } from "svelte";
  import RunSidebar from "./lib/RunSidebar.svelte";
  import WorkspaceSection from "./lib/WorkspaceSection.svelte";
  import {
    addPanelToWorkspace,
    createSection,
    loadWorkspaceSections,
    reorderSection as reorderSectionList,
    saveWorkspaceSections,
  } from "./lib/workspace.js";
  import {
    dataRefreshKey,
    loadRunVisibility,
    saveRunVisibility,
    syncRunVisibility,
    visibleRuns as filterVisibleRuns,
  } from "./lib/runGroups.js";
  import {
    fetchManifest,
    fetchRuns,
    fetchWorkspaceDocument,
    saveWorkspaceDocument,
  } from "./lib/api.js";
  import { MAX_PANEL_CATALOG_KEYS } from "./lib/catalogLimits.js";
  import { fetchCatalogCached, fetchMetricFamiliesCached } from "./lib/metricsCache.js";
  import MetricFamilySidebar from "./lib/MetricFamilySidebar.svelte";
  import {
    panelPartialForMetricFamily,
    panelVizForMetricFamily,
  } from "./lib/sidebarMetrics.js";

  const POLL_INTERVAL_MS = 10_000;

  let theme = $state("light");
  let runs = $state([]);
  let focusedRunId = $state("");
  let runVisibility = $state(loadRunVisibility());
  let manifest = $state({});
  let metrics = $state([]);
  /** @type {import('./lib/sidebarMetrics.js').MetricFamily[]} */
  let metricFamilies = $state([]);
  let sections = $state([]);
  /** @type {string} W&B scan root for this server instance (workspace identity). */
  let workspaceScanRoot = $state("");
  let workspaceHydrated = $state(false);
  let draggingSectionId = $state(null);
  let dropTargetSectionId = $state(null);

  function serializeSections(list) {
    return JSON.parse(JSON.stringify(list));
  }

  async function hydrateWorkspace() {
    try {
      const document = await fetchWorkspaceDocument();
      workspaceScanRoot = document.scan_root ?? "";
      if (document.sections?.length) {
        sections = document.sections.map((section) => createSection(section));
      } else {
        const local = loadWorkspaceSections();
        if (local.length) {
          sections = local;
          await saveWorkspaceDocument(serializeSections(sections));
        }
      }
    } catch {
      const local = loadWorkspaceSections();
      if (local.length) {
        sections = local;
      }
    } finally {
      workspaceHydrated = true;
    }
  }

  const runsShown = $derived(filterVisibleRuns(runs, runVisibility));

  const focusedRun = $derived(
    runs.find((run) => run.wandb_run_id === focusedRunId) ?? null,
  );

  let chartRefreshKey = $state("");

  function buildChartRefreshKey(runList, visibility, focusId) {
    const visible = filterVisibleRuns(runList, visibility);
    let key = dataRefreshKey(visible);
    if (!focusId) {
      return key;
    }
    const focused = runList.find((run) => run.wandb_run_id === focusId);
    return `${key}|focus:${focusId}:${focused?.store_version ?? 0}`;
  }

  function syncChartRefreshKey(runList = runs) {
    const next = buildChartRefreshKey(runList, runVisibility, focusedRunId);
    if (next !== chartRefreshKey) {
      chartRefreshKey = next;
    }
  }

  async function loadFocusedRunData(runId) {
    if (!runId) {
      manifest = {};
      metrics = [];
      metricFamilies = [];
      return;
    }
    manifest = await fetchManifest(runId);
    const version = manifest.store_version ?? 0;
    const familiesPayload = await fetchMetricFamiliesCached(runId, version);
    metricFamilies = familiesPayload.families ?? [];
    const count = manifest.metric_count ?? 0;
    if (count > MAX_PANEL_CATALOG_KEYS) {
      metrics = [];
    } else {
      const catalog = await fetchCatalogCached(runId, version);
      metrics = catalog.metrics ?? [];
    }
    syncChartRefreshKey(runs);
  }

  async function loadRuns() {
    runs = await fetchRuns();
    runVisibility = syncRunVisibility(runVisibility, runs);
    if (!focusedRunId && runs.length) {
      focusedRunId = runs[0].wandb_run_id;
    }
    syncChartRefreshKey(runs);
    if (focusedRunId) {
      await loadFocusedRunData(focusedRunId);
    }
  }

  async function poll() {
    const nextRuns = await fetchRuns();
    const previousKey = chartRefreshKey;
    runVisibility = syncRunVisibility(runVisibility, nextRuns);
    if (!focusedRunId && nextRuns.length) {
      focusedRunId = nextRuns[0].wandb_run_id;
    }
    runs = nextRuns;
    const nextKey = buildChartRefreshKey(nextRuns, runVisibility, focusedRunId);
    if (!focusedRunId) {
      if (nextKey !== previousKey) {
        chartRefreshKey = nextKey;
      }
      return;
    }
    const nextManifest = await fetchManifest(focusedRunId);
    const storeChanged = nextManifest.store_version !== manifest.store_version;
    manifest = nextManifest;
    if (storeChanged) {
      const version = nextManifest.store_version ?? 0;
      const familiesPayload = await fetchMetricFamiliesCached(focusedRunId, version);
      metricFamilies = familiesPayload.families ?? [];
      const count = nextManifest.metric_count ?? 0;
      if (count > MAX_PANEL_CATALOG_KEYS) {
        metrics = [];
      } else {
        const catalog = await fetchCatalogCached(focusedRunId, version);
        metrics = catalog.metrics ?? [];
      }
    }
    if (nextKey !== previousKey || storeChanged) {
      chartRefreshKey = buildChartRefreshKey(nextRuns, runVisibility, focusedRunId);
    }
  }

  function addSection() {
    sections = [...sections, createSection({ name: `Section ${sections.length + 1}` })];
  }

  function addPanelForMetricFamily(family) {
    const viz = panelVizForMetricFamily(family);
    const partial = panelPartialForMetricFamily(family);
    sections = addPanelToWorkspace(sections, viz, partial);
  }

  function removeSection(sectionId) {
    sections = sections.filter((section) => section.id !== sectionId);
  }

  function clearSectionDrag() {
    draggingSectionId = null;
    dropTargetSectionId = null;
  }

  function onSectionDragStart(sectionId) {
    draggingSectionId = sectionId;
    dropTargetSectionId = null;
  }

  function onSectionDragOver(sectionId) {
    if (!draggingSectionId || draggingSectionId === sectionId) {
      return;
    }
    dropTargetSectionId = sectionId;
  }

  function onSectionDrop(targetSectionId) {
    if (!draggingSectionId) {
      return;
    }
    sections = reorderSectionList(sections, draggingSectionId, targetSectionId);
    clearSectionDrag();
  }

  $effect(() => {
    document.documentElement.dataset.theme = theme;
  });

  $effect(() => {
    if (!workspaceHydrated) {
      return;
    }
    saveWorkspaceSections(sections);
    const snapshot = serializeSections(sections);
    const timer = setTimeout(() => {
      void saveWorkspaceDocument(snapshot).catch(() => {});
    }, 400);
    return () => clearTimeout(timer);
  });

  $effect(() => {
    saveRunVisibility(runVisibility);
  });

  onMount(() => {
    void hydrateWorkspace().then(() => loadRuns());
    let timer = setInterval(poll, POLL_INTERVAL_MS);

    function onVisibilityChange() {
      clearInterval(timer);
      if (document.visibilityState === "visible") {
        poll();
        timer = setInterval(poll, POLL_INTERVAL_MS);
      }
    }

    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  });

  $effect(() => {
    focusedRunId;
    runVisibility;
    syncChartRefreshKey(runs);
  });

  const plottableMetrics = $derived(
    metrics.filter((metric) => metric.dtype === "float" || metric.dtype === "int"),
  );
</script>

<div class="layout">
  <aside class="sidebar">
    <h1 style="font-size:1.1rem;margin:0 0 0.75rem;">BEV</h1>
    <p style="font-size:0.8rem;color:var(--muted);margin:0 0 0.75rem;">
      Runs grouped by <strong>display name</strong>. <strong>Click a run</strong> to focus the
      workspace (multipanels, heatmaps, metric tree). Checkboxes add runs to <strong>line</strong>
      charts for comparison.
    </p>
    <label style="font-size:0.85rem;display:block;margin-bottom:0.5rem;">
      Theme
      <select bind:value={theme} style="width:100%;margin-top:0.25rem;">
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </label>
    <h2 style="font-size:0.9rem;margin:1rem 0 0.5rem;">Runs</h2>
    <RunSidebar
      {runs}
      bind:visibility={runVisibility}
      bind:focusedRunId
      onfocusrun={loadFocusedRunData}
    />
    <h2 style="font-size:0.9rem;margin:1rem 0 0.5rem;">Metrics</h2>
    <p style="font-size:0.75rem;color:var(--muted);margin:0 0 0.35rem;">
      Scoped <code>section/metric</code> families from the focused run. Click to add a
      <strong>line</strong> (single series) or <strong>multipanel</strong> (layer/architecture
      breakdown). Use panel settings to drill into paths.
    </p>
    <MetricFamilySidebar class="metric-tree" families={metricFamilies} onSelectFamily={addPanelForMetricFamily} />
  </aside>

  <main class="main">
    <div class="topbar">
      <div>
        {#if workspaceScanRoot}
          <div
            style="font-size:0.75rem;color:var(--muted);margin-bottom:0.2rem;"
            title="Workspace (W&B scan root)"
          >
            {workspaceScanRoot}
          </div>
        {/if}
        <strong>{focusedRun?.display_name ?? "Select a run"}</strong>
        {#if manifest.project}
          <span style="color:var(--muted);"> · {manifest.project}</span>
        {/if}
        {#if runsShown.length}
          <span style="color:var(--muted); font-size:0.85rem;">
            · {runsShown.length} run{runsShown.length === 1 ? "" : "s"} on charts
          </span>
        {/if}
      </div>
      <span class="badge">
        {#if focusedRun}
          focus {focusedRun.wandb_run_id} · step {manifest.last_step ?? "—"}
        {:else}
          no run focused
        {/if}
      </span>
    </div>

    <div class="dashboard-toolbar">
      <button
        type="button"
        class="toolbar-btn toolbar-btn-primary"
        data-testid="workspace-add-section"
        onclick={addSection}
      >
        Add section
      </button>
    </div>

    {#if sections.length === 0}
      <div class="placeholder dashboard-empty">
        No sections yet. Click <strong>Add section</strong>, then add panels inside it (like W&B
        workspaces).
      </div>
    {:else}
      <div class="workspace-sections" role="list" aria-label="Workspace sections">
      {#each sections as section, index (section.id)}
        <WorkspaceSection
          bind:section={sections[index]}
          visibleRuns={runsShown}
          {focusedRun}
          {focusedRunId}
          dataRefreshKey={chartRefreshKey}
          {plottableMetrics}
          {metricFamilies}
          isDragging={draggingSectionId === section.id}
          isDropTarget={dropTargetSectionId === section.id &&
            draggingSectionId !== section.id}
          onsectiondragstart={onSectionDragStart}
          onsectiondragend={clearSectionDrag}
          onsectiondragover={onSectionDragOver}
          onsectiondrop={onSectionDrop}
          onremovesection={removeSection}
        />
      {/each}
      </div>
    {/if}
  </main>
</div>
